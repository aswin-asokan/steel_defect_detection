// defect_detection_screen.dart
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dashrectpainter.dart';

class DefectDetectionScreen extends StatefulWidget {
  const DefectDetectionScreen({super.key});

  @override
  State<DefectDetectionScreen> createState() => _DefectDetectionScreenState();
}

class _DefectDetectionScreenState extends State<DefectDetectionScreen>
    with SingleTickerProviderStateMixin {
  Uint8List? _selectedImageBytes;
  String? _overlayImage;
  String? _binaryMask;
  bool _isLoading = false;

  // Snapshot polling live view
  Uint8List? _liveFrameBytes;
  bool _liveDefect = false;
  // ignore: unused_field
  bool _prevLiveDefect = false;
  Timer? _pollTimer;

  final picker = ImagePicker();
  final String apiHost = "http://127.0.0.1:5000";
  String get apiUrl => "$apiHost/predict";
  String get snapshotUrl => "$apiHost/snapshot";

  late final TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    // start polling when "Live" tab is active — we'll start/stop in didChangeDependencies or listen to tab
    _tabController.addListener(_handleTabChange);
  }

  void _handleTabChange() {
    if (_tabController.index == 1) {
      startPollingSnapshots();
    } else {
      stopPollingSnapshots();
    }
  }

  @override
  void dispose() {
    stopPollingSnapshots();
    _tabController.removeListener(_handleTabChange);
    _tabController.dispose();
    super.dispose();
  }

  Future<void> pickImage() async {
    try {
      final pickedFile = await picker.pickImage(source: ImageSource.gallery);
      if (pickedFile == null) return;
      final bytes = await pickedFile.readAsBytes();
      setState(() {
        _selectedImageBytes = bytes;
        _overlayImage = null;
        _binaryMask = null;
      });
    } catch (e) {
      debugPrint("Image picking failed: $e");
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Failed to pick image: $e")));
      }
    }
  }

  Future<void> uploadImage() async {
    if (_selectedImageBytes == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Please select an image first.")),
        );
      }
      return;
    }

    setState(() => _isLoading = true);
    try {
      final uri = Uri.parse(apiUrl);
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          _selectedImageBytes!,
          filename: "upload.jpg",
        ),
      );
      final response = await request.send();
      final respStr = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final data = json.decode(respStr);
        setState(() {
          _overlayImage = data['overlay_image'];
          _binaryMask = data['binary_mask'];
        });
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Error: ${response.statusCode}")),
          );
        }
      }
    } catch (e) {
      debugPrint("Upload error: $e");
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Upload failed: $e")));
      }
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Polling loop: GET snapshot JSON and update UI
  void startPollingSnapshots({int intervalMs = 200}) {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(Duration(milliseconds: intervalMs), (_) async {
      try {
        final uri = Uri.parse(
          snapshotUrl + "?t=${DateTime.now().millisecondsSinceEpoch}",
        );
        final resp = await http.get(uri).timeout(const Duration(seconds: 2));
        if (resp.statusCode == 200) {
          final data = json.decode(resp.body);
          if (data['status'] == 'success' && data['image'] != null) {
            final String imgData = data['image'];
            // image expected like "data:image/jpeg;base64,...." or just base64
            final base = imgData.contains(',')
                ? imgData.split(',').last
                : imgData;
            final bytes = base64Decode(base);
            final bool defect = data['defect'] == true;

            if (mounted) {
              setState(() {
                _liveFrameBytes = bytes;
                _liveDefect = defect;
              });
              _prevLiveDefect = _liveDefect;
            }
          }
        }
      } catch (e) {
        // ignore transient failures
      }
    });
  }

  void stopPollingSnapshots() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  Widget _buildImagePickerCard() {
    return GestureDetector(
      onTap: pickImage,
      child: Container(
        width: double.infinity,
        height: 350,
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
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const SizedBox(height: 30),
                        Icon(
                          Icons.upload_file_outlined,
                          size: 80,
                          color: const Color(0xffc5c5c5),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          "Tap to upload image",
                          style: GoogleFonts.urbanist(
                            fontWeight: FontWeight.w700,
                            fontSize: 20,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          "Supports JPG, PNG and other image formats",
                          style: GoogleFonts.urbanist(
                            fontWeight: FontWeight.w400,
                            fontSize: 16,
                          ),
                        ),
                        const SizedBox(height: 30),
                      ],
                    ),
                  )
                : Image.memory(
                    _selectedImageBytes!,
                    fit: BoxFit.cover,
                    width: double.infinity,
                    height: double.infinity,
                  ),
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
        minimumSize: const Size(double.infinity, 60),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
      child: _isLoading
          ? Text(
              "Processing...",
              style: GoogleFonts.urbanist(
                fontWeight: FontWeight.w400,
                fontSize: 18,
                color: Colors.white,
              ),
            )
          : Text(
              "Start Analysis",
              style: GoogleFonts.urbanist(
                fontWeight: FontWeight.w400,
                fontSize: 18,
                color: Colors.white,
              ),
            ),
    );
  }

  Widget _buildResultsColumn() {
    return ListView(
      children: [
        const SizedBox(height: 40),
        if (_overlayImage != null) ...[
          Text(
            "Defect Overlay:",
            style: GoogleFonts.urbanist(
              fontWeight: FontWeight.w600,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.memory(
              base64Decode(_overlayImage!.split(',').last),
              fit: BoxFit.contain,
            ),
          ),
          const SizedBox(height: 20),
        ],
        if (_binaryMask != null) ...[
          Text(
            "Binary Mask:",
            style: GoogleFonts.urbanist(
              fontWeight: FontWeight.w600,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.memory(
              base64Decode(_binaryMask!.split(',').last),
              fit: BoxFit.contain,
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildLiveStreamView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 12),
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
                                child: Row(
                                  children: [
                                    const Icon(
                                      Icons.warning_amber_rounded,
                                      color: Colors.white,
                                      size: 18,
                                    ),
                                    const SizedBox(width: 6),
                                    Text(
                                      "Defect",
                                      style: GoogleFonts.urbanist(
                                        color: Colors.white,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                        ],
                      ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 12),
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
            Icon(Icons.construction_outlined, color: Colors.blue, size: 40),
            const SizedBox(width: 12),
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
            // Image tab
            Row(
              children: [
                Expanded(
                  flex: 5,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const SizedBox(height: 10),
                      Text(
                        "Analyze New Image",
                        style: GoogleFonts.urbanist(
                          fontWeight: FontWeight.w700,
                          fontSize: 22,
                        ),
                      ),
                      const SizedBox(height: 12),
                      _buildImagePickerCard(),
                      const SizedBox(height: 14),
                      _buildUploadButton(),
                    ],
                  ),
                ),
                const SizedBox(width: 18),
                Expanded(
                  flex: 5,
                  child: (_overlayImage == null && _binaryMask == null)
                      ? Center(
                          child: Text(
                            "Upload to start image analysis",
                            style: GoogleFonts.urbanist(
                              fontWeight: FontWeight.w700,
                              fontSize: 18,
                            ),
                          ),
                        )
                      : _buildResultsColumn(),
                ),
              ],
            ),
            // Live tab
            _buildLiveStreamView(),
          ],
        ),
      ),
    );
  }
}
