import AppKit
import Foundation

let destination = URL(fileURLWithPath: CommandLine.arguments[1])
let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".iconset")
try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: temporary) }
func polygon(_ points: [NSPoint], fill: NSColor) {
    let path = NSBezierPath()
    path.move(to: points[0])
    for point in points.dropFirst() { path.line(to: point) }
    path.close()
    fill.setFill()
    path.fill()
    NSColor(calibratedRed: 0.52, green: 0.93, blue: 1, alpha: 0.9).setStroke()
    path.lineWidth = 10
    path.lineJoinStyle = .round
    path.stroke()
}
func render(_ pixels: Int, name: String) throws {
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
    let transform = NSAffineTransform()
    transform.scale(by: CGFloat(pixels)/1024)
    transform.concat()
    let rect = NSRect(x: 64, y: 64, width: 896, height: 896)
    let background = NSBezierPath(roundedRect: rect, xRadius: 205, yRadius: 205)
    NSGradient(starting: NSColor(calibratedRed: 0.15, green: 0.21, blue: 0.28, alpha: 1), ending: NSColor(calibratedRed: 0.045, green: 0.07, blue: 0.11, alpha: 1))!.draw(in: background, angle: -80)
    polygon([NSPoint(x: 512,y: 790), NSPoint(x: 760,y: 650), NSPoint(x: 512,y: 502), NSPoint(x: 264,y: 650)], fill: NSColor(calibratedRed: 0.18, green: 0.77, blue: 0.88, alpha: 1))
    polygon([NSPoint(x: 264,y: 650), NSPoint(x: 512,y: 502), NSPoint(x: 512,y: 216), NSPoint(x: 264,y: 360)], fill: NSColor(calibratedRed: 0.09, green: 0.43, blue: 0.57, alpha: 1))
    polygon([NSPoint(x: 512,y: 502), NSPoint(x: 760,y: 650), NSPoint(x: 760,y: 360), NSPoint(x: 512,y: 216)], fill: NSColor(calibratedRed: 0.12, green: 0.61, blue: 0.72, alpha: 1))
    NSGraphicsContext.restoreGraphicsState()
    try bitmap.representation(using: .png, properties: [:])!.write(to: temporary.appendingPathComponent(name))
}
for size in [16, 32, 128, 256, 512] {
    try render(size, name: "icon_\(size)x\(size).png")
    try render(size*2, name: "icon_\(size)x\(size)@2x.png")
}
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", temporary.path, "-o", destination.path]
try process.run()
process.waitUntilExit()
if process.terminationStatus != 0 { exit(process.terminationStatus) }
