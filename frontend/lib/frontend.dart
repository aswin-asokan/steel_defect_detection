import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;

class DefectDetectionScreen extends StatefulWidget {
  const DefectDetectionScreen({super.key});

  @override
  State<DefectDetectionScreen> createState() => _DefectDetectionScreenState();
}

class _DefectDetectionScreenState extends State<DefectDetectionScreen> {
  // Set local for laptop backend camera mode, or set ngrok URL for mobile-camera upload mode.
  final String apiHost = "http://127.0.0.1:5000";

  bool get _useLocalBackendCamera {
    final host = Uri.tryParse(apiHost)?.host.toLowerCase();
    return host == '127.0.0.1' || host == 'localhost';
  }

  String get predictUrl => "$apiHost/predict";
  String get snapshotUrl => "$apiHost/snapshot";
  String get switchUrl => "$apiHost/switch";
  String get switchOptionsUrl => "$apiHost/switch/options";
  String get logsCurrentUrl => "$apiHost/logs/current";
  String get patternLatestUrl => "$apiHost/pattern/latest";
  String get patternManualUrl => "$apiHost/pattern/manual";

  List<Map<String, String>> _switchOptions = const [
    {"id": "sam", "label": "SAM"},
    {"id": "mobilesam", "label": "mobileSAM"},
    {"id": "yolo26", "label": "yolo26"},
    {"id": "yolo26_mobilesam", "label": "yolo26+mobileSAM"},
  ];

  String _liveOption = "yolo26_mobilesam";

  CameraController? _cameraController;
  bool _cameraReady = false;
  bool _processing = false;

  Timer? _captureTimer;
  Timer? _snapshotTimer;
  int _metaTick = 0;

  Uint8List? _processedFrameBytes;
  Map<String, dynamic>? _lastResult;
  Map<String, dynamic>? _logsInfo;
  Map<String, dynamic>? _patternSummary;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await _fetchSwitchOptions();
    if (_useLocalBackendCamera) {
      await _switchBackendMode(_liveOption, showSnack: false);
      _startSnapshotLoop();
    } else {
      await _initCamera();
      _startCaptureLoop();
    }
    await _refreshMeta();
  }

  @override
  void dispose() {
    _captureTimer?.cancel();
    _snapshotTimer?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _fetchSwitchOptions() async {
    try {
      final resp = await http
          .get(Uri.parse(switchOptionsUrl))
          .timeout(const Duration(seconds: 3));
      if (resp.statusCode != 200) return;
      final data = json.decode(resp.body) as Map<String, dynamic>;
      final opts = (data["options"] as List<dynamic>? ?? []);
      if (opts.isEmpty) return;

      final parsed = opts
          .map((o) {
            final m = o as Map<String, dynamic>;
            return {
              "id": (m["id"] ?? "").toString(),
              "label": (m["label"] ?? m["id"] ?? "").toString(),
            };
          })
          .where((e) => (e["id"] ?? "").isNotEmpty)
          .toList();

      if (!mounted || parsed.isEmpty) return;
      setState(() {
        _switchOptions = parsed;
        if (!_switchOptions.any((e) => e["id"] == _liveOption)) {
          _liveOption = _switchOptions.first["id"]!;
        }
      });
    } catch (_) {}
  }

  Future<void> _switchBackendMode(
    String option, {
    bool showSnack = true,
  }) async {
    try {
      final resp = await http
          .post(
            Uri.parse(switchUrl),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'option': option}),
          )
          .timeout(const Duration(seconds: 4));
      final data = json.decode(resp.body) as Map<String, dynamic>;
      if (resp.statusCode != 200 || data['status'] != 'success') {
        throw Exception((data['message'] ?? 'Switch failed').toString());
      }
      if (showSnack && mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Switched to $option')));
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Switch failed: $e')));
    }
  }

  Future<void> _initCamera() async {
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) return;
      final cam = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cams.first,
      );
      final controller = CameraController(
        cam,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted) return;
      setState(() {
        _cameraController = controller;
        _cameraReady = true;
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Camera init failed: $e')));
    }
  }

  Uint8List? _decodeDataUri(String? value) {
    if (value == null || value.isEmpty) return null;
    try {
      final base = value.contains(',') ? value.split(',').last : value;
      return base64Decode(base);
    } catch (_) {
      return null;
    }
  }

  void _startSnapshotLoop() {
    _snapshotTimer?.cancel();
    _snapshotTimer = Timer.periodic(const Duration(milliseconds: 180), (
      _,
    ) async {
      await _pullSnapshot();
      _metaTick++;
      if (_metaTick % 20 == 0) {
        await _refreshMeta();
      }
    });
  }

  Future<void> _pullSnapshot() async {
    try {
      final resp = await http
          .get(
            Uri.parse(
              "$snapshotUrl?t=${DateTime.now().millisecondsSinceEpoch}",
            ),
          )
          .timeout(const Duration(seconds: 2));
      if (resp.statusCode != 200) return;
      final data = json.decode(resp.body) as Map<String, dynamic>;
      if (data['status'] != 'success') return;
      final img = _decodeDataUri(data['image']?.toString());
      if (!mounted) return;
      setState(() {
        if (img != null) _processedFrameBytes = img;
        _lastResult = data;
      });
    } catch (_) {}
  }

  void _startCaptureLoop() {
    _captureTimer?.cancel();
    _captureTimer = Timer.periodic(const Duration(milliseconds: 700), (
      _,
    ) async {
      await _captureAndPredict();
      _metaTick++;
      if (_metaTick % 8 == 0) {
        await _refreshMeta();
      }
    });
  }

  Future<void> _captureAndPredict() async {
    final controller = _cameraController;
    if (controller == null || !_cameraReady || _processing) return;
    if (!controller.value.isInitialized || controller.value.isTakingPicture) {
      return;
    }

    _processing = true;
    try {
      final shot = await controller.takePicture();
      final bytes = await shot.readAsBytes();

      final req = http.MultipartRequest('POST', Uri.parse(predictUrl));
      req.fields['option'] = _liveOption;
      req.files.add(
        http.MultipartFile.fromBytes('image', bytes, filename: 'frame.jpg'),
      );

      final resp = await req.send();
      final text = await resp.stream.bytesToString();
      final data = json.decode(text) as Map<String, dynamic>;

      if (resp.statusCode == 200 && data['status'] == 'success') {
        final overlay = _decodeDataUri(data['overlay_image']?.toString());
        if (!mounted) return;
        setState(() {
          _lastResult = data;
          if (overlay != null) _processedFrameBytes = overlay;
        });
      }
    } catch (_) {
      // transient capture/network failures are ignored for live loop
    } finally {
      _processing = false;
    }
  }

  Future<void> _refreshMeta() async {
    await Future.wait([_fetchLogs(), _fetchPatternLatest()]);
  }

  Future<void> _fetchLogs() async {
    try {
      final resp = await http
          .get(Uri.parse(logsCurrentUrl))
          .timeout(const Duration(seconds: 3));
      if (resp.statusCode != 200) return;
      final data = json.decode(resp.body) as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _logsInfo = data);
    } catch (_) {}
  }

  Future<void> _fetchPatternLatest() async {
    try {
      final resp = await http
          .get(Uri.parse(patternLatestUrl))
          .timeout(const Duration(seconds: 3));
      if (resp.statusCode != 200) {
        if (mounted) setState(() => _patternSummary = null);
        return;
      }
      final data = json.decode(resp.body) as Map<String, dynamic>;
      if (!mounted) return;
      setState(
        () => _patternSummary = data['summary'] as Map<String, dynamic>?,
      );
    } catch (_) {}
  }

  Future<void> _runPatternNow() async {
    try {
      final resp = await http
          .post(
            Uri.parse(patternManualUrl),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({}),
          )
          .timeout(const Duration(seconds: 10));
      final data = json.decode(resp.body) as Map<String, dynamic>;
      if (resp.statusCode == 200 && data['status'] == 'success') {
        if (!mounted) return;
        setState(
          () => _patternSummary = data['summary'] as Map<String, dynamic>?,
        );
      }
    } catch (_) {}
  }

  Widget _modelSwitcher() {
    return DropdownButtonFormField<String>(
      initialValue: _liveOption,
      onChanged: (value) async {
        if (value == null) return;
        setState(() => _liveOption = value);
        if (_useLocalBackendCamera) {
          await _switchBackendMode(value);
        }
      },
      items: _switchOptions
          .map(
            (o) => DropdownMenuItem<String>(
              value: o['id'],
              child: Text(o['label'] ?? o['id']!),
            ),
          )
          .toList(),
      decoration: InputDecoration(
        labelText: 'Live Method',
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

  Widget _liveAlertCard() {
    final result = _lastResult;
    final defect = result?['defect'] == true;
    final mode = (result?['mode'] ?? '-').toString();
    final fps = (result?['fps'] ?? 0).toString();
    final types = ((result?['defect_types'] as List?) ?? [])
        .map((e) => e.toString())
        .join(', ');

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: defect ? Colors.red.shade50 : Colors.green.shade50,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            defect ? 'Defect detected' : 'No defect detected',
            style: GoogleFonts.urbanist(fontWeight: FontWeight.w700),
          ),
          Text('Mode: $mode', style: GoogleFonts.urbanist()),
          Text(
            'Types: ${types.isEmpty ? '-' : types}',
            style: GoogleFonts.urbanist(),
          ),
          Text('Backend FPS: $fps', style: GoogleFonts.urbanist()),
          Text(
            _useLocalBackendCamera
                ? 'Source: Backend local camera'
                : 'Source: Mobile camera upload',
            style: GoogleFonts.urbanist(fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }

  Widget _logsAndPatternCard() {
    final csv = (_logsInfo?['csv_log'] ?? '-').toString();
    final jsn = (_logsInfo?['json_log'] ?? '-').toString();
    final top = ((_patternSummary?['top_defect_types'] as List?) ?? [])
        .map(
          (e) => e is Map<String, dynamic>
              ? '${e['defect_type']} (${e['count']})'
              : e.toString(),
        )
        .join(', ');

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Logs',
            style: GoogleFonts.urbanist(fontWeight: FontWeight.w700),
          ),
          Text('CSV: $csv', style: GoogleFonts.urbanist(fontSize: 12)),
          Text('JSON: $jsn', style: GoogleFonts.urbanist(fontSize: 12)),
          const SizedBox(height: 8),
          Row(
            children: [
              Text(
                'Pattern',
                style: GoogleFonts.urbanist(fontWeight: FontWeight.w700),
              ),
              const Spacer(),
              ElevatedButton(
                onPressed: _runPatternNow,
                child: const Text('Run now'),
              ),
            ],
          ),
          Text(
            top.isEmpty ? 'No pattern summary yet' : 'Top types: $top',
            style: GoogleFonts.urbanist(fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _videoPanels() {
    if (_useLocalBackendCamera) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Container(
          color: Colors.black,
          child: _processedFrameBytes == null
              ? const Center(child: CircularProgressIndicator())
              : Image.memory(_processedFrameBytes!, fit: BoxFit.contain),
        ),
      );
    }

    return Row(
      children: [
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Container(
              color: Colors.black,
              child: !_cameraReady || _cameraController == null
                  ? const Center(child: CircularProgressIndicator())
                  : CameraPreview(_cameraController!),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Container(
              color: Colors.black,
              child: _processedFrameBytes == null
                  ? const Center(child: Text('Waiting for processed frames...'))
                  : Image.memory(_processedFrameBytes!, fit: BoxFit.contain),
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF4F3F3),
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: Text(
          _useLocalBackendCamera
              ? 'Steel Defect Detection - Local Backend Camera'
              : 'Steel Defect Detection - Mobile Camera',
          style: GoogleFonts.urbanist(fontWeight: FontWeight.w700),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _modelSwitcher(),
            const SizedBox(height: 10),
            _liveAlertCard(),
            const SizedBox(height: 10),
            Expanded(child: _videoPanels()),
            const SizedBox(height: 10),
            _logsAndPatternCard(),
          ],
        ),
      ),
    );
  }
}
