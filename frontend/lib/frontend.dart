import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import 'dashrectpainter.dart';

class DefectDetectionScreen extends StatefulWidget {
  const DefectDetectionScreen({super.key});

  @override
  State<DefectDetectionScreen> createState() => _DefectDetectionScreenState();
}

class _DefectDetectionScreenState extends State<DefectDetectionScreen>
    with SingleTickerProviderStateMixin {
  final picker = ImagePicker();
  final String apiHost = "http://127.0.0.1:5000";

  String get snapshotUrl => "$apiHost/snapshot";
  String get predictUrl => "$apiHost/predict";
  String get switchUrl => "$apiHost/switch";
  String get switchOptionsUrl => "$apiHost/switch/options";
  String get logsCurrentUrl => "$apiHost/logs/current";
  String get patternLatestUrl => "$apiHost/pattern/latest";
  String get patternManualUrl => "$apiHost/pattern/manual";

  late final TabController _tabController;

  List<Map<String, String>> _switchOptions = const [
    {"id": "sam", "label": "SAM"},
    {"id": "mobilesam", "label": "mobileSAM"},
    {"id": "yolo26", "label": "yolo26"},
    {"id": "yolo26_mobilesam", "label": "mobileSAM+yolo26"},
    {
      "id": "preprocess_yolo26_mobilesam",
      "label": "preprocess+yolo26+mobileSAM",
    },
  ];

  String _imageOption = "yolo26_mobilesam";
  String _liveOption = "yolo26_mobilesam";
  bool _switchingLive = false;

  Uint8List? _selectedImageBytes;
  String? _overlayImage;
  String? _binaryMask;
  Map<String, dynamic>? _imageResult;
  bool _isLoading = false;

  Uint8List? _liveFrameBytes;
  Map<String, dynamic>? _liveSnapshot;
  bool _liveDefect = false;
  Timer? _pollTimer;
  int _pollTick = 0;

  Map<String, dynamic>? _logsInfo;
  Map<String, dynamic>? _patternSummary;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _tabController.addListener(_handleTabChange);
    _fetchSwitchOptions();
  }

  @override
  void dispose() {
    stopPollingSnapshots();
    _tabController.removeListener(_handleTabChange);
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _fetchSwitchOptions() async {
    try {
      final resp = await http
          .get(Uri.parse(switchOptionsUrl))
          .timeout(const Duration(seconds: 3));
      if (resp.statusCode != 200) return;
      final data = json.decode(resp.body);
      final List<dynamic> opts = (data["options"] as List<dynamic>? ?? []);
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

      if (parsed.isEmpty) return;
      if (!mounted) return;
      setState(() {
        _switchOptions = parsed;
        if (!_switchOptions.any((e) => e["id"] == _imageOption)) {
          _imageOption = _switchOptions.first["id"]!;
        }
        if (!_switchOptions.any((e) => e["id"] == _liveOption)) {
          _liveOption = _switchOptions.first["id"]!;
        }
      });
    } catch (_) {}
  }

  void _handleTabChange() {
    if (_tabController.index == 1) {
      _applyLiveSwitch(_liveOption, showSnack: false);
      startPollingSnapshots();
      _refreshLiveMeta();
    } else {
      stopPollingSnapshots();
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

  Future<void> pickImage() async {
    try {
      final pickedFile = await picker.pickImage(source: ImageSource.gallery);
      if (pickedFile == null) return;
      final bytes = await pickedFile.readAsBytes();
      if (!mounted) return;
      setState(() {
        _selectedImageBytes = bytes;
        _overlayImage = null;
        _binaryMask = null;
        _imageResult = null;
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Failed to pick image: $e")));
    }
  }

  Future<void> uploadImage() async {
    if (_selectedImageBytes == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please select an image first.")),
      );
      return;
    }

    setState(() => _isLoading = true);
    try {
      final request = http.MultipartRequest('POST', Uri.parse(predictUrl));
      request.fields['option'] = _imageOption;
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          _selectedImageBytes!,
          filename: 'upload.jpg',
        ),
      );

      final response = await request.send();
      final text = await response.stream.bytesToString();
      final data = json.decode(text);

      if (response.statusCode == 200 && data['status'] == 'success') {
        if (!mounted) return;
        setState(() {
          _overlayImage = data['overlay_image'] as String?;
          _binaryMask = data['binary_mask'] as String?;
          _imageResult = (data as Map<String, dynamic>);
        });
      } else {
        final msg = data is Map<String, dynamic>
            ? (data['message'] ?? 'Image prediction failed').toString()
            : 'Image prediction failed';
        if (!mounted) return;
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(msg)));
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Upload failed: $e")));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _applyLiveSwitch(String option, {bool showSnack = true}) async {
    setState(() => _switchingLive = true);
    try {
      final resp = await http
          .post(
            Uri.parse(switchUrl),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'option': option}),
          )
          .timeout(const Duration(seconds: 5));
      final data = json.decode(resp.body);
      if (resp.statusCode != 200 || data['status'] != 'success') {
        throw Exception((data['message'] ?? 'Switch failed').toString());
      }
      if (showSnack && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Live mode switched to $option')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Switch failed: $e')));
      }
    } finally {
      if (mounted) setState(() => _switchingLive = false);
    }
  }

  void startPollingSnapshots({int intervalMs = 220}) {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(Duration(milliseconds: intervalMs), (_) async {
      await _pollSnapshot();
      _pollTick++;
      if (_pollTick % 20 == 0) {
        await _refreshLiveMeta();
      }
    });
  }

  Future<void> _pollSnapshot() async {
    try {
      final uri = Uri.parse(
        "$snapshotUrl?t=${DateTime.now().millisecondsSinceEpoch}",
      );
      final resp = await http.get(uri).timeout(const Duration(seconds: 2));
      if (resp.statusCode != 200) return;
      final data = json.decode(resp.body);
      if (data['status'] != 'success') return;

      final bytes = _decodeDataUri(data['image']?.toString());
      if (bytes == null) return;

      if (!mounted) return;
      setState(() {
        _liveFrameBytes = bytes;
        _liveSnapshot = (data as Map<String, dynamic>);
        _liveDefect = data['defect'] == true;
      });
    } catch (_) {}
  }

  Future<void> _refreshLiveMeta() async {
    await Future.wait([_fetchLogs(), _fetchPatternLatest()]);
  }

  Future<void> _fetchLogs() async {
    try {
      final resp = await http
          .get(Uri.parse(logsCurrentUrl))
          .timeout(const Duration(seconds: 3));
      if (resp.statusCode != 200) return;
      final data = json.decode(resp.body);
      if (!mounted) return;
      setState(() {
        _logsInfo = data as Map<String, dynamic>;
      });
    } catch (_) {}
  }

  Future<void> _fetchPatternLatest() async {
    try {
      final resp = await http
          .get(Uri.parse(patternLatestUrl))
          .timeout(const Duration(seconds: 3));
      if (resp.statusCode != 200) {
        if (mounted) {
          setState(() => _patternSummary = null);
        }
        return;
      }
      final data = json.decode(resp.body) as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _patternSummary = data['summary'] as Map<String, dynamic>?;
      });
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
        setState(() {
          _patternSummary = data['summary'] as Map<String, dynamic>?;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Pattern detection completed')),
        );
      } else {
        throw Exception(
          (data['message'] ?? 'Pattern detection failed').toString(),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Pattern run failed: $e')));
    }
  }

  void stopPollingSnapshots() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  Widget _modelSwitcher({
    required String value,
    required ValueChanged<String?> onChanged,
    required String title,
    bool enabled = true,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.urbanist(
            fontWeight: FontWeight.w700,
            fontSize: 15,
          ),
        ),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          initialValue: value,
          onChanged: enabled ? onChanged : null,
          items: _switchOptions
              .map(
                (o) => DropdownMenuItem<String>(
                  value: o['id'],
                  child: Text(o['label'] ?? o['id']!),
                ),
              )
              .toList(),
          decoration: InputDecoration(
            filled: true,
            fillColor: Colors.white,
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
            contentPadding: const EdgeInsets.symmetric(
              horizontal: 12,
              vertical: 10,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildImagePickerCard() {
    return GestureDetector(
      onTap: pickImage,
      child: Container(
        width: double.infinity,
        height: 320,
        decoration: BoxDecoration(borderRadius: BorderRadius.circular(12)),
        child: CustomPaint(
          painter: _selectedImageBytes == null
              ? DashRectPainter(
                  color: Colors.grey.shade400,
                  strokeWidth: 1.5,
                  dashLength: 10,
                  gapLength: 5,
                  borderRadius: 12,
                )
              : null,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: _selectedImageBytes == null
                ? Center(
                    child: Text(
                      "Tap to upload image",
                      style: GoogleFonts.urbanist(
                        fontWeight: FontWeight.w700,
                        fontSize: 20,
                      ),
                    ),
                  )
                : Image.memory(_selectedImageBytes!, fit: BoxFit.cover),
          ),
        ),
      ),
    );
  }

  Widget _buildUploadButton() {
    return ElevatedButton(
      onPressed: _isLoading
          ? null
          : () async {
              if (_selectedImageBytes == null) {
                await pickImage();
              } else {
                await uploadImage();
              }
            },
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.blue,
        minimumSize: const Size(double.infinity, 54),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
      child: Text(
        _isLoading ? "Processing..." : "Start Analysis",
        style: GoogleFonts.urbanist(
          fontWeight: FontWeight.w600,
          fontSize: 18,
          color: Colors.white,
        ),
      ),
    );
  }

  Widget _buildImageResultInfo() {
    if (_imageResult == null) {
      return Text(
        "Upload to start image analysis",
        style: GoogleFonts.urbanist(fontWeight: FontWeight.w700, fontSize: 18),
      );
    }

    final defect = _imageResult!['defect'] == true;
    final fps = (_imageResult!['fps'] ?? 0).toString();
    final mode = (_imageResult!['mode'] ?? '-').toString();
    final types = ((_imageResult!['defect_types'] as List?) ?? [])
        .map((e) => e.toString())
        .join(', ');

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: defect ? Colors.red.shade50 : Colors.green.shade50,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Mode: $mode',
            style: GoogleFonts.urbanist(fontWeight: FontWeight.w600),
          ),
          Text(
            'Defect: ${defect ? 'YES' : 'NO'}',
            style: GoogleFonts.urbanist(fontWeight: FontWeight.w700),
          ),
          Text(
            'Types: ${types.isEmpty ? '-' : types}',
            style: GoogleFonts.urbanist(),
          ),
          Text('FPS: $fps', style: GoogleFonts.urbanist()),
        ],
      ),
    );
  }

  Widget _buildResultsColumn() {
    final overlayBytes = _decodeDataUri(_overlayImage);
    final maskBytes = _decodeDataUri(_binaryMask);

    return ListView(
      children: [
        _buildImageResultInfo(),
        const SizedBox(height: 16),
        if (overlayBytes != null) ...[
          Text(
            "Overlay",
            style: GoogleFonts.urbanist(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.memory(overlayBytes, fit: BoxFit.contain),
          ),
          const SizedBox(height: 16),
        ],
        if (maskBytes != null) ...[
          Text(
            "Binary Mask",
            style: GoogleFonts.urbanist(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.memory(maskBytes, fit: BoxFit.contain),
          ),
        ],
      ],
    );
  }

  Widget _buildLiveAlertCard() {
    final snap = _liveSnapshot;
    final defect = snap?['defect'] == true;
    final fps = (snap?['fps'] ?? 0).toString();
    final mode = (snap?['mode'] ?? '-').toString();
    final types = ((snap?['defect_types'] as List?) ?? [])
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
          Row(
            children: [
              Icon(
                defect ? Icons.warning_amber_rounded : Icons.check_circle,
                color: defect ? Colors.red : Colors.green,
              ),
              const SizedBox(width: 8),
              Text(
                defect ? 'Defect detected' : 'No defect detected',
                style: GoogleFonts.urbanist(fontWeight: FontWeight.w700),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text('Mode: $mode', style: GoogleFonts.urbanist()),
          Text(
            'Types: ${types.isEmpty ? '-' : types}',
            style: GoogleFonts.urbanist(),
          ),
          Text('FPS: $fps', style: GoogleFonts.urbanist()),
        ],
      ),
    );
  }

  Widget _buildLogsAndPattern() {
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
          const SizedBox(height: 6),
          Text('CSV: $csv', style: GoogleFonts.urbanist(fontSize: 12)),
          Text('JSON: $jsn', style: GoogleFonts.urbanist(fontSize: 12)),
          const SizedBox(height: 10),
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
          const SizedBox(height: 6),
          Text(
            top.isEmpty ? 'No pattern summary yet' : 'Top types: $top',
            style: GoogleFonts.urbanist(fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildLiveStreamView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _modelSwitcher(
          value: _liveOption,
          enabled: !_switchingLive,
          title: 'Live Method',
          onChanged: (value) async {
            if (value == null) return;
            setState(() => _liveOption = value);
            await _applyLiveSwitch(value);
          },
        ),
        const SizedBox(height: 10),
        _buildLiveAlertCard(),
        const SizedBox(height: 10),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Container(
              color: Colors.black,
              child: Center(
                child: _liveFrameBytes == null
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: const [
                          CircularProgressIndicator(),
                          SizedBox(height: 12),
                          Text("Waiting for frames..."),
                        ],
                      )
                    : Stack(
                        children: [
                          Positioned.fill(
                            child: Image.memory(
                              _liveFrameBytes!,
                              fit: BoxFit.contain,
                            ),
                          ),
                          if (_liveDefect)
                            Positioned(
                              top: 16,
                              right: 16,
                              child: Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 8,
                                  vertical: 6,
                                ),
                                decoration: BoxDecoration(
                                  color: Colors.redAccent,
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                child: Text(
                                  'Defect Alert',
                                  style: GoogleFonts.urbanist(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ),
                            ),
                        ],
                      ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 10),
        _buildLogsAndPattern(),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF4F3F3),
      appBar: AppBar(
        scrolledUnderElevation: 0,
        backgroundColor: Colors.white,
        title: Row(
          children: [
            const Icon(
              Icons.construction_outlined,
              color: Colors.blue,
              size: 36,
            ),
            const SizedBox(width: 10),
            Text(
              "Steel Defect Detection",
              style: GoogleFonts.urbanist(
                fontWeight: FontWeight.w700,
                fontSize: 24,
              ),
            ),
          ],
        ),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: "Image"),
            Tab(text: "Live"),
          ],
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: TabBarView(
          controller: _tabController,
          children: [
            Row(
              children: [
                Expanded(
                  flex: 5,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _modelSwitcher(
                        value: _imageOption,
                        title: 'Image Method',
                        onChanged: (value) {
                          if (value == null) return;
                          setState(() => _imageOption = value);
                        },
                      ),
                      const SizedBox(height: 12),
                      _buildImagePickerCard(),
                      const SizedBox(height: 12),
                      _buildUploadButton(),
                    ],
                  ),
                ),
                const SizedBox(width: 18),
                Expanded(flex: 5, child: _buildResultsColumn()),
              ],
            ),
            _buildLiveStreamView(),
          ],
        ),
      ),
    );
  }
}
