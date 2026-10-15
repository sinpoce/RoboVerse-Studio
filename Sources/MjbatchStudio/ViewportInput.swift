// SPDX-License-Identifier: Apache-2.0

import AppKit
import SwiftUI

/// Scroll-wheel / trackpad-scroll events and double-clicks over the viewport.
/// A local event monitor handles events only while the pointer is over the
/// workbench viewport, leaving inspector controls and other windows untouched.
final class ScrollWheelInput: ObservableObject {
    weak var probe: NSView?
    /// The event, the cursor as a fraction of the viewport (x right, y down,
    /// centre = 0) and the viewport's size in points.
    var onScroll: ((NSEvent, CGPoint, CGSize) -> Void)?
    /// A double-click on the picture (not on a control drawn over it).
    var onDoubleClick: (() -> Void)?
    private var monitor: Any?

    func start() {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.scrollWheel, .leftMouseDown]) { [weak self] event in
            guard let self, let probe = self.probe, event.window === probe.window else { return event }
            let p = probe.convert(event.locationInWindow, from: nil)
            let b = probe.bounds
            guard b.contains(p), b.width > 0, b.height > 0 else { return event }
            if event.type == .leftMouseDown {
                if event.clickCount == 2, let content = probe.window?.contentView,
                   !(content.hitTest(content.convert(event.locationInWindow, from: nil)) is NSControl) {
                    self.onDoubleClick?()
                }
                return event  // the drag gesture still gets it
            }
            let y = probe.isFlipped ? p.y / b.height : 1 - p.y / b.height
            self.onScroll?(event, CGPoint(x: p.x / b.width - 0.5, y: y - 0.5), b.size)
            return nil  // consumed: nothing behind the viewport scrolls
        }
    }

    func stop() {
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
    }

    deinit { stop() }
}

struct ScrollWheelProbe: NSViewRepresentable {
    let input: ScrollWheelInput

    /// Only measures where the viewport is; never the target of a click, so
    /// SwiftUI's gestures (drag, double-click) get the mouse untouched.
    final class Probe: NSView {
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }

    func makeNSView(context: Context) -> NSView {
        let view = Probe()
        input.probe = view
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        input.probe = view
    }
}
