from __future__ import annotations


APP_NAME = "三角洲全自动卡邮件"


DARK_GREEN_THEME = """
QWidget {
    background: #070b10;
    color: #d7e4e1;
    font-family: "Microsoft YaHei UI";
}

QMainWindow, QDialog {
    background: #070b10;
}

QWidget#appRoot {
    background: #070b10;
}







QFrame#sponsorBoard {
    background: #0b1419;
    border: 1px solid #294047;
    border-left: 3px solid #3df1bb;
}

QLabel#sponsorDescription {
    color: #a9c3bc;
    padding: 8px 0;
}





QLabel#brandTitle {
    color: #3df1bb;
    font-size: 16pt;
    font-weight: 700;
    padding-right: 10px;
}

QLabel#fieldLabel, QLabel#runtimeState {
    color: #8fa5a5;
}

QLabel#sectionHeading {
    color: #64e9c0;
    font-weight: 700;
    padding: 4px 0 6px 0;
    border-bottom: 1px solid #1b2b32;
}

QPushButton#previewToggleButton {
    min-width: 64px;
    max-width: 64px;
    min-height: 28px;
    max-height: 28px;
    padding: 0 8px;
}

QLabel#workspaceTitle {
    color: #e7f5f1;
    font-size: 15pt;
    font-weight: 700;
    padding: 2px 0 5px 0;
}

QFrame#workspaceSidebar {
    background: #0b1218;
    border: 1px solid #1d2c34;
}

QLabel#sidebarHeading {
    background: transparent;
    color: #79918f;
    font-weight: 700;
    padding: 2px 8px 7px 8px;
}

QLabel#tutorialSubtitle {
    color: #90a8a5;
    padding: 0 0 4px 0;
}

QLabel#tutorialTitleHint {
    color: #718785;
    font-size: 11pt;
}

QTabWidget#tutorialTabs::pane {
    border: 1px solid #20353a;
    background: #080e13;
    top: -1px;
}

QTabWidget#tutorialTabs QTabBar::tab {
    background: #0d171d;
    border: 1px solid #20353a;
    color: #92aaa6;
    min-width: 126px;
    min-height: 34px;
    padding: 2px 10px;
}

QTabWidget#tutorialTabs QTabBar::tab:hover {
    color: #d8f3eb;
    border-color: #3b7d6e;
}

QTabWidget#tutorialTabs QTabBar::tab:selected {
    background: #10382f;
    border-color: #3b9e83;
    color: #e5fff7;
    font-weight: 700;
}

QWidget#tutorialPageContent {
    background: #080e13;
}

QFrame#tutorialIntro {
    background: #0b1b1a;
    border: 1px solid #214b42;
    border-left: 3px solid #3df1bb;
}

QLabel#tutorialIntroTitle {
    background: transparent;
    color: #7cf0ce;
    font-size: 12pt;
    font-weight: 700;
}

QLabel#tutorialIntroBody {
    background: transparent;
    color: #a9bfbb;
}

QFrame#tutorialSection, QFrame#tutorialWarningSection {
    background: #0b1419;
    border: 1px solid #20353a;
}

QFrame#tutorialWarningSection {
    border-left: 3px solid #d6a134;
}

QLabel#tutorialStepBadge {
    background: #145342;
    border: 1px solid #55c8aa;
    color: #edfff9;
    font-weight: 700;
}

QLabel#tutorialSectionTitle {
    background: transparent;
    color: #e3f3ef;
    font-size: 11pt;
    font-weight: 700;
}

QLabel#tutorialBody {
    background: transparent;
    color: #cbdcd8;
    line-height: 1.45;
}

QLabel#tutorialTrialNotice, QLabel#tutorialMarketNotice {
    background: transparent;
    color: #cbdcd8;
    line-height: 1.45;
}

QLabel#tutorialPathLabel {
    background: transparent;
    color: #82a39e;
    font-weight: 700;
    padding-top: 3px;
}

QLabel#tutorialPath {
    background: #070c10;
    border: 1px solid #294047;
    color: #8cebcf;
    font-family: "Cascadia Mono", "Microsoft YaHei UI";
    font-size: 9pt;
    padding: 8px;
}

QPushButton#workspaceNavButton {
    background: transparent;
    border: 1px solid transparent;
    color: #a8bab7;
    min-height: 36px;
    padding: 3px 10px;
    text-align: left;
}

QPushButton#workspaceNavButton:hover {
    background: #122129;
    border-color: #294047;
}

QPushButton#workspaceNavButton:checked {
    background: #10382f;
    border-color: #3b9e83;
    color: #e5fff7;
    font-weight: 700;
}

QLabel#statusMessage {
    color: #54e9bb;
    font-weight: 700;
    padding: 3px 0;
}

QFrame#announcementBanner {
    background: #0b1b1a;
    border: 1px solid #214b42;
    border-left: 3px solid #3df1bb;
}

QFrame#mailIntegrationBand {
    background: #0b1419;
    border: 1px solid #20353a;
    border-left: 3px solid #29c99a;
}

QFrame#mailRunStatus {
    background: #0b1419;
    border: 1px solid #20353a;
    border-left: 3px solid #29c99a;
}

QScrollArea#mailWorkspacePage {
    background: transparent;
    border: 0;
}

QLabel#mailIntegrationTitle {
    background: transparent;
    color: #7cf0ce;
    font-weight: 700;
}

QLabel#mailIntegrationState {
    background: transparent;
    color: #f3c861;
    font-weight: 700;
}

QLabel#mailIntegrationProgress, QLabel#mailIntegrationResult,
QLabel#mailIntegrationDetail, QLabel#settingHint {
    background: transparent;
    color: #9db2b1;
}

QLabel#hotkeyHint {
    background: transparent;
    color: #718785;
    font-size: 8pt;
}

QLabel#mailBindingStatus, QLabel#mailPakStatus {
    background: transparent;
    color: #9db2b1;
}

QLabel#announcementText {
    background: transparent;
    color: #bde8dc;
    font-weight: 600;
}

QPushButton#announcementAction {
    background: #10382f;
    border-color: #3b9e83;
    color: #dffff5;
    font-weight: 700;
}

QPushButton#announcementAction:hover {
    background: #145342;
    border-color: #55edc2;
}

QToolButton#helpButton {
    min-width: 28px;
    max-width: 28px;
    padding: 0;
    background: #10382f;
    border-color: #3b9e83;
    color: #dffff5;
    font-weight: 700;
}

QToolButton#helpButton:hover {
    background: #145342;
    border-color: #55edc2;
}

QLabel#noticeTitle {
    color: #3df1bb;
    font-size: 18pt;
    font-weight: 700;
}

QLabel#noticeSubtitle {
    color: #8fa5a5;
    padding-bottom: 4px;
}

QFrame#noticePanel {
    background: #0b1419;
    border: 1px solid #20353a;
    border-left: 3px solid #29c99a;
}

QLabel#noticeSectionHeading {
    background: transparent;
    color: #7cf0ce;
    font-size: 12pt;
    font-weight: 700;
    padding-bottom: 5px;
}

QLabel#noticeBody {
    background: transparent;
    color: #d2e3df;
}

QLabel#noticeLink {
    background: transparent;
    color: #55edc2;
}

QLabel#noticeLink:hover {
    color: #b8ffeb;
}

QPushButton, QToolButton {
    background: #101a20;
    border: 1px solid #253841;
    color: #d8e7e4;
    min-height: 28px;
    padding: 2px 12px;
}

QPushButton:hover, QToolButton:hover {
    background: #17262d;
    border-color: #3ce5ba;
}

QPushButton:pressed, QToolButton:pressed {
    background: #0a4236;
}

QPushButton:disabled, QToolButton:disabled {
    background: #0b1116;
    border-color: #18242b;
    color: #52615f;
}

QPushButton:checked {
    background: #0c5544;
    border-color: #3df1bb;
    color: #e1fff6;
}

QPushButton#primaryAction {
    background: #1bc995;
    border-color: #4ef6c4;
    color: #04150f;
    font-weight: 700;
}

QPushButton#primaryAction:hover {
    background: #4ae7b8;
}

QPushButton#warningAction {
    background: #201a0b;
    border-color: #9b7017;
    color: #f3c861;
}

QPushButton#warningAction:hover {
    background: #34280d;
    border-color: #e4aa2a;
}

QPushButton#dangerAction {
    color: #ff988f;
    border-color: #743b38;
}

QPushButton#dangerAction:hover {
    background: #321818;
    border-color: #f1746d;
}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTimeEdit, QDateEdit, QTextEdit,
QTableWidget, QAbstractScrollArea {
    background: #0b1218;
    border: 1px solid #1d2c34;
    color: #d7e4e1;
    selection-background-color: #0d5e4b;
    selection-color: #effff9;
}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTimeEdit, QDateEdit {
    min-height: 28px;
    padding: 1px 8px;
}

QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QTimeEdit:hover, QDateEdit:hover {
    border-color: #45d8b1;
}

QComboBox::drop-down {
    border-left: 1px solid #1d2c34;
    width: 24px;
}

QComboBox QAbstractItemView {
    background: #0c151b;
    border: 1px solid #31505a;
    selection-background-color: #0d5e4b;
}

QTableWidget {
    alternate-background-color: #0d151c;
    gridline-color: #17262d;
}

QTableWidget::item {
    padding: 3px 6px;
    border-bottom: 1px solid #132129;
}

QTableWidget::item:selected {
    background: #103f38;
    color: #e9fff8;
}

QHeaderView::section {
    background: #0e181e;
    border: 0;
    border-right: 1px solid #1c2d34;
    border-bottom: 1px solid #35505a;
    color: #88f0cf;
    font-weight: 700;
    padding: 6px;
}

QTableCornerButton::section {
    background: #0e181e;
    border: 0;
    border-right: 1px solid #1c2d34;
    border-bottom: 1px solid #35505a;
}

QTextEdit {
    font-family: "Cascadia Mono", "Microsoft YaHei UI";
    line-height: 1.35;
}

QLabel#livePreview {
    background: #05090d;
    border: 1px solid #1c3437;
    color: #8ba6a1;
}

QSplitter::handle {
    background: #14242a;
}

QSplitter::handle:hover {
    background: #1cc99a;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #42616a;
    background: #0a1116;
}

QCheckBox::indicator:hover {
    border-color: #3df1bb;
}

QCheckBox::indicator:checked {
    background: #1bc995;
    border-color: #5bffd0;
}

QScrollBar:vertical {
    background: #080d12;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #28434a;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background: #3a7670;
}

QScrollBar:horizontal {
    background: #080d12;
    height: 10px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #28434a;
    min-width: 28px;
}

QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
    border: none;
}

QToolTip {
    background: #122026;
    border: 1px solid #3ba990;
    color: #e7fff7;
    padding: 5px;
}

QDialogButtonBox {
    border-top: 1px solid #1d3035;
    padding-top: 8px;
}
"""
