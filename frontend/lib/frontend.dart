import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:frontend/dashrectpainter.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class DefectDetectionScreen extends StatefulWidget {
  const DefectDetectionScreen({Key? key}) : super(key: key);

  @override
  State<DefectDetectionScreen> createState() => _DefectDetectionScreenState();
}

class _DefectDetectionScreenState extends State<DefectDetectionScreen> {
  Uint8List? _selectedImageBytes;
  String? _overlayImage;
  String? _binaryMask;
  bool _isLoading = false;

  final picker = ImagePicker();

  /// ⚠️ IMPORTANT: Use your machine’s **LAN IP**, not localhost, for web.
  /// Example: "http://192.168.1.5:5000/predict"
  final String apiUrl = "http://127.0.0.1:5000/predict";

  Future<void> pickImage() async {
    try {
      final pickedFile = await picker.pickImage(source: ImageSource.gallery);
      if (pickedFile == null) return;

      // On web, `readAsBytes()` works fine
      final bytes = await pickedFile.readAsBytes();

      setState(() {
        _selectedImageBytes = bytes;
        _overlayImage = null;
        _binaryMask = null;
      });
    } catch (e) {
      debugPrint("Image picking failed: $e");
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Failed to pick image: $e")));
    }
  }

  Future<void> uploadImage() async {
    if (_selectedImageBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please select an image first.")),
      );
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
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error: ${response.statusCode}")),
        );
      }
    } catch (e) {
      debugPrint("Upload error: $e");
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Upload failed: $e")));
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF4F3F3),
      appBar: AppBar(
        scrolledUnderElevation: 0,
        backgroundColor: const Color(0xffffffff),
        title: Row(
          spacing: 15,
          children: [
            Icon(Icons.construction_outlined, color: Colors.blue, size: 40),
            Text(
              "Steel Defect Detection",
              style: GoogleFonts.urbanist(
                fontWeight: FontWeight.w700,
                fontSize: 30,
              ),
            ),
          ],
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          spacing: 18,
          children: [
            Expanded(
              flex: 5,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                spacing: 18,
                children: [
                  const SizedBox(height: 10),
                  Text(
                    "Analyze New Image",
                    style: GoogleFonts.urbanist(
                      fontWeight: FontWeight.w700,
                      fontSize: 25,
                    ),
                  ),
                  // Image picker
                  GestureDetector(
                    onTap: pickImage,
                    child: Container(
                      width: double.infinity,
                      height: 350,
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
                                      const SizedBox(height: 50),
                                      Icon(
                                        Icons.upload_file_outlined,
                                        size: 80,
                                        color: const Color(0xffc5c5c5),
                                      ),
                                      Text(
                                        "Tap to upload image",
                                        style: GoogleFonts.urbanist(
                                          fontWeight: FontWeight.w700,
                                          fontSize: 20,
                                        ),
                                      ),
                                      Text(
                                        "Supports JPG, PNG and other image formats",
                                        style: GoogleFonts.urbanist(
                                          fontWeight: FontWeight.w400,
                                          fontSize: 18,
                                        ),
                                      ),
                                      const SizedBox(height: 50),
                                    ],
                                  ),
                                )
                              : Image.memory(
                                  _selectedImageBytes!,
                                  fit: BoxFit.cover,
                                ),
                        ),
                      ),
                    ),
                  ),

                  // Upload button
                  ElevatedButton(
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
                      backgroundColor: Colors.blue, // Sets the blue color
                      minimumSize: const Size(
                        double.infinity,
                        60,
                      ), // Increased height
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(
                          12,
                        ), // Creates the rounded corners
                      ),
                    ),
                    child:
                        _isLoading // Changed from 'label' to 'child'
                        ? Text(
                            "Processing...",
                            style: GoogleFonts.urbanist(
                              fontWeight: FontWeight.w400,
                              fontSize: 18,
                              color: const Color(0xffffffff),
                            ),
                          )
                        : Text(
                            "Start Analysis",
                            style: GoogleFonts.urbanist(
                              fontWeight: FontWeight.w400,
                              fontSize: 18,
                              color: const Color(0xffffffff),
                            ),
                          ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 10),
            if (_overlayImage == null || _binaryMask == null)
              Expanded(
                flex: 5,
                child: Center(
                  child: Text(
                    "Upload to start image analysis",
                    style: GoogleFonts.urbanist(
                      fontWeight: FontWeight.w700,
                      fontSize: 18,
                    ),
                  ),
                ),
              ),
            // Display results
            if (_overlayImage != null || _binaryMask != null)
              Expanded(
                flex: 5,
                child: ListView(
                  children: [
                    const SizedBox(height: 50),
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
                ),
              ),
          ],
        ),
      ),
    );
  }
}
