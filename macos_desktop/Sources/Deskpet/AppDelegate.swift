import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var controller: PetController?
    private var statusItem: NSStatusItem?
    private let context: PetContext

    init(context: PetContext) {
        self.context = context
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        controller = PetController()
        controller?.start()
        installStatusItem()
    }

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            button.image = NSImage(
                systemSymbolName: "pawprint.fill",
                accessibilityDescription: "Deskpet: \(PetConfig.petName) (\(context.session))"
            )
        }

        let menu = NSMenu()

        // Pet name (bold header)
        let header = NSMenuItem()
        header.attributedTitle = NSAttributedString(
            string: PetConfig.petName,
            attributes: [
                .font: NSFont.systemFont(ofSize: NSFont.systemFontSize, weight: .bold),
                .foregroundColor: NSColor.labelColor,
            ]
        )
        header.isEnabled = false
        menu.addItem(header)

        menu.addItem(infoItem(key: "session", value: context.session))
        menu.addItem(infoItem(key: "cwd", value: collapseHome(context.pwd)))

        menu.addItem(NSMenuItem.separator())

        let quit = NSMenuItem(
            title: "Quit Deskpet",
            action: #selector(quit),
            keyEquivalent: "q"
        )
        quit.target = self
        menu.addItem(quit)

        item.menu = menu
        statusItem = item
    }

    /// Disabled "key:  value" info row. Monospaced so fields align.
    private func infoItem(key: String, value: String) -> NSMenuItem {
        let padded = key.padding(toLength: 8, withPad: " ", startingAt: 0)
        let item = NSMenuItem()
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(
                ofSize: NSFont.smallSystemFontSize,
                weight: .regular
            ),
            .foregroundColor: NSColor.secondaryLabelColor,
        ]
        item.attributedTitle = NSAttributedString(
            string: "\(padded)\(value)",
            attributes: attrs
        )
        item.toolTip = value
        item.isEnabled = false
        return item
    }

    /// Replaces the $HOME prefix with ~ for friendlier display.
    private func collapseHome(_ path: String) -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        if path == home { return "~" }
        if path.hasPrefix(home + "/") {
            return "~" + path.dropFirst(home.count)
        }
        return path
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }
}
