// PoseExtractor.swift

import AVFoundation
import Vision
import CoreImage
import AppKit

// --- Structures for JSON Output (COCO-WholeBody format) ---
struct FrameData: Codable {
    let frame_index: Int
    // Structure: "keypoints": [[x, y], [x, y], ...], "scores": [s1, s2, ...]
    // COCO-WholeBody has 133 keypoints.
    let keypoints: [[Float]] // 133 x 2
    let scores: [Float]      // 133
}

struct VideoPoseData: Codable {
    let video_name: String
    let width: Int
    let height: Int
    let frames: [FrameData]
}

// Main struct
@main
struct PoseExtractor {
    static func main() async {
        let arguments = CommandLine.arguments
        // Usage: executable <input_video> <output_video> <output_json>
        guard arguments.count >= 3 else {
            print("Usage: \(arguments[0]) <input_video_path> <output_video_path> [output_json_path]")
            return
        }
        let inputURL = URL(fileURLWithPath: arguments[1])
        let outputURL = URL(fileURLWithPath: arguments[2])
        let jsonURL = arguments.count > 3 ? URL(fileURLWithPath: arguments[3]) : nil

        print("Starting UniSign-optimized pose extraction...")
        do {
            let poseData = try await processVideo(input: inputURL, output: outputURL)
            
            if let jsonURL = jsonURL, let poseData = poseData {
                print("Saving JSON to \(jsonURL.path)...")
                let encoder = JSONEncoder()
                // encoder.outputFormatting = .prettyPrinted // Uncomment for debugging, keep off for file size
                let data = try encoder.encode(poseData)
                try data.write(to: jsonURL)
                print("✅ JSON saved.")
            }
            print("✅ Video saved to: \(outputURL.path)")
        } catch {
            print("❌ An error occurred: \(error.localizedDescription)")
        }
    }
}

func processVideo(input inputURL: URL, output outputURL: URL) async throws -> VideoPoseData? {
    let asset = AVURLAsset(url: inputURL)
    guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
        throw NSError(domain: "PoseExtractor", code: 1, userInfo: [NSLocalizedDescriptionKey: "No video track found."])
    }
    
    let assetReader = try AVAssetReader(asset: asset)
    let readerOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ])
    assetReader.add(readerOutput)

    if FileManager.default.fileExists(atPath: outputURL.path) {
        try FileManager.default.removeItem(at: outputURL)
    }

    let assetWriter = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
    let videoSize = try await videoTrack.load(.naturalSize)
    let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: videoSize.width,
        AVVideoHeightKey: videoSize.height
    ])
    let pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: writerInput, sourcePixelBufferAttributes: nil)
    assetWriter.add(writerInput)

    guard assetReader.startReading(), assetWriter.startWriting() else {
        throw NSError(domain: "PoseExtractor", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to start reading/writing."])
    }
    assetWriter.startSession(atSourceTime: .zero)
    
    let visionQueue = DispatchQueue(label: "com.unisign.visionqueue", qos: .userInteractive)
    var recordedFrames: [FrameData] = []
    var frameCount = 0
    
    try await withCheckedThrowingContinuation { continuation in
        writerInput.requestMediaDataWhenReady(on: visionQueue) {
            while writerInput.isReadyForMoreMediaData {
                if let sampleBuffer = readerOutput.copyNextSampleBuffer() {
                    if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                        // Detect
                        let (processedBuffer, frameData) = detectPose(on: pixelBuffer, frameSize: videoSize, frameIndex: frameCount)
                        if let data = frameData {
                            recordedFrames.append(data)
                        }
                        frameCount += 1
                        
                        // Write (Optional: Pass through processed buffer if drawing is enabled)
                        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                        pixelBufferAdaptor.append(processedBuffer, withPresentationTime: presentationTime)
                    }
                } else {
                    writerInput.markAsFinished()
                    Task {
                        await assetWriter.finishWriting()
                        assetReader.cancelReading()
                        continuation.resume()
                    }
                    break
                }
            }
        }
    }
    
    return VideoPoseData(video_name: inputURL.lastPathComponent, width: Int(videoSize.width), height: Int(videoSize.height), frames: recordedFrames)
}

func detectPose(on pixelBuffer: CVPixelBuffer, frameSize: CGSize, frameIndex: Int) -> (CVPixelBuffer, FrameData?) {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let handler = VNImageRequestHandler(ciImage: ciImage, orientation: .up, options: [:])
    
    let bodyRequest = VNDetectHumanBodyPoseRequest()
    let handRequest = VNDetectHumanHandPoseRequest()
    handRequest.maximumHandCount = 2
    let faceRequest = VNDetectFaceLandmarksRequest()

    do {
        // Perform all requests in a single pass to leverage ANE optimization
        try handler.perform([bodyRequest, handRequest, faceRequest])
        
        // Extract Data using Fusion Strategy
        let frameData = extractFusedFrameData(
            body: bodyRequest.results?.first,
            hands: handRequest.results,
            faces: faceRequest.results?.first, // Assuming single person focus
            frameIndex: frameIndex,
            frameSize: frameSize
        )
        
        // Draw for visual verification
        let drawnBuffer = drawAnnotations(on: pixelBuffer,
                                          body: bodyRequest.results?.first,
                                          hands: handRequest.results,
                                          faces: faceRequest.results,
                                          frameSize: frameSize)
        
        return (drawnBuffer, frameData)
        
    } catch {
        print("Vision request failed: \(error)")
        return (pixelBuffer, nil)
    }
}

// --- Fusion & Mapping Logic (Vision -> COCO-WholeBody 133) ---

func extractFusedFrameData(body: VNHumanBodyPoseObservation?, hands: [VNHumanHandPoseObservation]?, faces: VNFaceObservation?, frameIndex: Int, frameSize: CGSize) -> FrameData {
    
    // Initialize storage
    var kps = Array(repeating: [Float(0.0), Float(0.0)], count: 133)
    var scores = Array(repeating: Float(0.0), count: 133)
    
    // Helper: Convert Normalized Vision Point (Bottom-Left origin) to Global Pixel (Top-Left origin)
    func normToGlobal(_ point: CGPoint) -> CGPoint {
        // Vision: x (0..1 left-right), y (0..1 bottom-top)
        // Target: x (0..W left-right), y (0..H top-bottom)
        return CGPoint(x: point.x * frameSize.width, y: (1.0 - point.y) * frameSize.height)
    }
    
    // Helper: Save point to arrays
    func setPt(_ index: Int, _ point: CGPoint, _ confidence: Float) {
        guard index >= 0 && index < 133 else { return }
        kps[index] = [Float(point.x), Float(point.y)]
        scores[index] = confidence
    }

    // --- 1. BODY MAPPING (Indices 0-16) ---
    // We map Vision keys to COCO indices.
    // Vision often lacks 0 (Nose), 1-4 (Eyes/Ears) on body request in older iOS,
    // but recent versions include them. If missing, we leave as 0 or could infer.
    // Here we map strictly what's available.
    
    var bodyWristLeft: CGPoint?
    var bodyWristRight: CGPoint?
    
    if let body = body, let points = try? body.recognizedPoints(.all) {
        let map: [(VNHumanBodyPoseObservation.JointName, Int)] = [
            (.nose, 0), (.leftEye, 1), (.rightEye, 2), (.leftEar, 3), (.rightEar, 4),
            (.leftShoulder, 5), (.rightShoulder, 6),
            (.leftElbow, 7), (.rightElbow, 8),
            (.leftWrist, 9), (.rightWrist, 10),
            (.leftHip, 11), (.rightHip, 12),
            (.leftKnee, 13), (.rightKnee, 14),
            (.leftAnkle, 15), (.rightAnkle, 16)
        ]
        
        for (joint, idx) in map {
            if let p = points[joint], p.confidence > 0.0 {
                let globalPt = normToGlobal(p.location)
                setPt(idx, globalPt, Float(p.confidence))
                
                // Capture wrists for Stitching
                if idx == 9 { bodyWristLeft = globalPt }
                if idx == 10 { bodyWristRight = globalPt }
            }
        }
    }
    
    // --- 2. FACE MAPPING (Indices 23-90) via Resampling ---
    // Vision contours -> Fixed 68 iBUG points
    if let face = faces, let landmarks = face.landmarks {
        let bbox = face.boundingBox
        let conf = Float(face.confidence)
        
        // Helper: Transform local face normalized point to global pixel
        func faceLocalToGlobal(_ pt: CGPoint) -> CGPoint {
            // pt is inside bbox (0..1) relative to bbox
            // bbox is inside image (0..1) bottom-left origin
            let globalNormX = bbox.origin.x + pt.x * bbox.size.width
            let globalNormY = bbox.origin.y + pt.y * bbox.size.height
            return normToGlobal(CGPoint(x: globalNormX, y: globalNormY))
        }
        
        // A. Jawline (23-39) - 17 points
        if let contour = landmarks.faceContour {
            let resampled = resample(points: contour.normalizedPoints, count: 17)
            for (i, pt) in resampled.enumerated() {
                setPt(23 + i, faceLocalToGlobal(pt), conf)
            }
        }
        
        // B. Eyebrows (Left: 40-44, Right: 45-49)
        if let lb = landmarks.leftEyebrow {
            let resampled = resample(points: lb.normalizedPoints, count: 5)
            for (i, pt) in resampled.enumerated() { setPt(40 + i, faceLocalToGlobal(pt), conf) }
        }
        if let rb = landmarks.rightEyebrow {
            let resampled = resample(points: rb.normalizedPoints, count: 5)
            for (i, pt) in resampled.enumerated() { setPt(45 + i, faceLocalToGlobal(pt), conf) }
        }
        
        // C. Nose (Bridge: 50-53, Base: 54-58)
        // Vision noseCrest gives bridge. Vision nose gives base/nostrils.
        if let crest = landmarks.noseCrest {
            let resampled = resample(points: crest.normalizedPoints, count: 4)
            for (i, pt) in resampled.enumerated() { setPt(50 + i, faceLocalToGlobal(pt), conf) }
        }
        if let base = landmarks.nose {
            let resampled = resample(points: base.normalizedPoints, count: 5)
            for (i, pt) in resampled.enumerated() { setPt(54 + i, faceLocalToGlobal(pt), conf) }
        }
        
        // D. Eyes (Left: 59-64, Right: 65-70) - 6 points each (ignoring pupil)
        if let le = landmarks.leftEye {
            let resampled = resample(points: le.normalizedPoints, count: 6)
            // Note: COCO/iBUG winding might differ from Apple. Apple usually clockwise.
            // iBUG starts corner. We blindly map resampled points.
            for (i, pt) in resampled.enumerated() { setPt(59 + i, faceLocalToGlobal(pt), conf) }
        }
        if let re = landmarks.rightEye {
            let resampled = resample(points: re.normalizedPoints, count: 6)
            for (i, pt) in resampled.enumerated() { setPt(65 + i, faceLocalToGlobal(pt), conf) }
        }
        
        // E. Mouth (Outer: 71-82 (12pts), Inner: 83-90 (8pts))
        if let outer = landmarks.outerLips {
            let resampled = resample(points: outer.normalizedPoints, count: 12)
            for (i, pt) in resampled.enumerated() { setPt(71 + i, faceLocalToGlobal(pt), conf) }
        }
        if let inner = landmarks.innerLips {
            let resampled = resample(points: inner.normalizedPoints, count: 8)
            for (i, pt) in resampled.enumerated() { setPt(83 + i, faceLocalToGlobal(pt), conf) }
        }
    }
    
    // --- 3. HAND MAPPING & STITCHING (Indices 91-132) ---
    // Left: 91-111, Right: 112-132
    
    if let hands = hands {
        for hand in hands {
            // Determine side.
            // Vision chirality is from person's perspective? Usually.
            // COCO Left Hand indices: 91-111.
            let isLeft = hand.chirality == .left
            let baseIndex = isLeft ? 91 : 112
            let bodyWrist = isLeft ? bodyWristLeft : bodyWristRight
            
            // Get all points first to calculate offset
            guard let points = try? hand.recognizedPoints(.all) else { continue }
            
            // Vision Hand Wrist
            guard let visionWristPt = points[.wrist], visionWristPt.confidence > 0 else { continue }
            let handWristGlobal = normToGlobal(visionWristPt.location)
            
            // Calculate Anchor Offset (Stitching)
            var offset = CGPoint.zero
            if let bodyW = bodyWrist {
                offset = CGPoint(x: bodyW.x - handWristGlobal.x, y: bodyW.y - handWristGlobal.y)
                // print("Stitching \(hand.chirality): Delta \(offset)") // Debug
            }
            
            let joints: [VNHumanHandPoseObservation.JointName] = [
                .wrist,
                .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
                .indexMCP, .indexPIP, .indexDIP, .indexTip,
                .middleMCP, .middlePIP, .middleDIP, .middleTip,
                .ringMCP, .ringPIP, .ringDIP, .ringTip,
                .littleMCP, .littlePIP, .littleDIP, .littleTip
            ]
            
            for (i, jName) in joints.enumerated() {
                if let p = points[jName], p.confidence > 0 {
                    let rawGlobal = normToGlobal(p.location)
                    let stitchedGlobal = CGPoint(x: rawGlobal.x + offset.x, y: rawGlobal.y + offset.y)
                    setPt(baseIndex + i, stitchedGlobal, Float(p.confidence))
                }
            }
        }
    }
    
    return FrameData(frame_index: frameIndex, keypoints: kps, scores: scores)
}

// Helper to resample array of points to target count (Simple Linear Interpolation)
func resample(points: [CGPoint], count: Int) -> [CGPoint] {
    guard points.count > 0 else { return [] }
    if points.count == count { return points }
    var result: [CGPoint] = []
    // We treat the points as a sequence. 
    // For indices 0 to count-1, we map to source index space.
    let maxSrcIdx = Double(points.count - 1)
    let maxDstIdx = Double(count - 1)
    
    for i in 0..<count {
        let percent = Double(i) / maxDstIdx
        let srcPos = percent * maxSrcIdx
        
        let idx1 = Int(floor(srcPos))
        let idx2 = min(idx1 + 1, points.count - 1)
        let t = CGFloat(srcPos - Double(idx1))
        
        let p1 = points[idx1]
        let p2 = points[idx2]
        
        // Linear blend
        let newX = p1.x + (p2.x - p1.x) * t
        let newY = p1.y + (p2.y - p1.y) * t
        result.append(CGPoint(x: newX, y: newY))
    }
    return result
}

// --- Drawing Logic ---
func drawAnnotations(on pixelBuffer: CVPixelBuffer,
                     body: VNHumanBodyPoseObservation?,
                     hands: [VNHumanHandPoseObservation]?,
                     faces: [VNFaceObservation]?,
                     frameSize: CGSize) -> CVPixelBuffer {
    
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return pixelBuffer }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    
    guard let context = CGContext(data: baseAddress,
                                  width: Int(frameSize.width),
                                  height: Int(frameSize.height),
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    else {
        return pixelBuffer
    }

    // NOTE: We want to draw on the image which is Top-Left origin (0,0 at top-left).
    // Vision points are Normalized Bottom-Left origin (0,0 at bottom-left).
    // To ensure WYSIWYG with our JSON export (which converts to Top-Left),
    // we will manually convert coordinates during drawing instead of flipping the context matrix.
    // This avoids confusion with VNImagePointForNormalizedPoint behavior.

    // context.translateBy(x: 0, y: frameSize.height) // REMOVED
    // context.scaleBy(x: 1.0, y: -1.0) // REMOVED

    if let body = body { drawBody(context: context, observation: body, frameSize: frameSize) }
    if let hands = hands { for hand in hands { drawHand(context: context, observation: hand, frameSize: frameSize) } }
    if let faces = faces { for face in faces { drawFace(context: context, observation: face, frameSize: frameSize) } }
    
    return pixelBuffer
}

// Helper for drawing: specific Vision -> Image Top-Left conversion
// UPDATE: For CVPixelBuffer/CoreGraphics context (usually Bottom-Left origin),
// we should NOT flip Y if we want to draw 'upright' relative to the image data?
// Actually, if context is Bottom-Left, and we want to draw at Vision Y (0.9, Top),
// we draw at 0.9 * H. 
// So we just pass through.
func visionToImage(_ point: CGPoint, _ w: Int, _ h: Int) -> CGPoint {
    return CGPoint(x: point.x * CGFloat(w), y: point.y * CGFloat(h))
}

func drawBody(context: CGContext, observation: VNHumanBodyPoseObservation, frameSize: CGSize) {
    guard let points = try? observation.recognizedPoints(.all) else { return }
    context.setStrokeColor(NSColor.green.cgColor); context.setLineWidth(4.0)
    let connections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
        (.neck, .root), (.leftShoulder, .rightShoulder), (.leftShoulder, .leftHip), (.rightShoulder, .rightHip), (.leftHip, .rightHip),
        (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist), (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist)
    ]
    let w = Int(frameSize.width)
    let h = Int(frameSize.height)
    
    for (start, end) in connections {
        guard let p1 = points[start], p1.confidence > 0.1, let p2 = points[end], p2.confidence > 0.1 else { continue }
        let pt1 = visionToImage(p1.location, w, h)
        let pt2 = visionToImage(p2.location, w, h)
        context.move(to: pt1); context.addLine(to: pt2); context.strokePath()
    }
}

func drawHand(context: CGContext, observation: VNHumanHandPoseObservation, frameSize: CGSize) {
    guard let points = try? observation.recognizedPoints(.all) else { return }
    context.setStrokeColor(NSColor.cyan.cgColor); context.setLineWidth(2.0)
    let fingers = [[.thumbCMC, .thumbMP, .thumbIP, .thumbTip], [.indexMCP, .indexPIP, .indexDIP, .indexTip], [.middleMCP, .middlePIP, .middleDIP, .middleTip], [.ringMCP, .ringPIP, .ringDIP, .ringTip], [.littleMCP, .littlePIP, .littleDIP, .littleTip]] as [[VNHumanHandPoseObservation.JointName]]
    let w = Int(frameSize.width)
    let h = Int(frameSize.height)

    for finger in fingers {
        for i in 0..<finger.count-1 {
            guard let p1 = points[finger[i]], p1.confidence > 0.1, let p2 = points[finger[i+1]], p2.confidence > 0.1 else { continue }
            let pt1 = visionToImage(p1.location, w, h)
            let pt2 = visionToImage(p2.location, w, h)
            context.move(to: pt1); context.addLine(to: pt2); context.strokePath()
        }
    }
}

func drawFace(context: CGContext, observation: VNFaceObservation, frameSize: CGSize) {
    context.setStrokeColor(NSColor.yellow.cgColor); context.setLineWidth(2.0)
    // Face landmarks are also normalized relative to the bounding box, but pointsInImage converts to Image Coords?
    // Vision documentation says pointsInImage returns coordinates in the image.
    // IMPORTANT: pointsInImage results often depend on how the request was handled (orientation).
    // Let's trust pointsInImage but we might need to flip Y if it returns Bottom-Left coords.
    // Usually pointsInImage returns Bottom-Left coords. So we need to flip Y.
    
    if let contour = observation.landmarks?.faceContour {
        let rawPts = contour.pointsInImage(imageSize: frameSize)
        if rawPts.count > 0 {
            let pts = rawPts.map { CGPoint(x: $0.x, y: frameSize.height - $0.y) }
            context.addLines(between: pts); context.strokePath()
        }
    }
}