// ====================================================
// ÇOKLU ENJEKSİYON KORUMASI
// ====================================================
window._mangaInitTime = Date.now();
const _myInitTime = window._mangaInitTime;
function isLatest() { return window._mangaInitTime === _myInitTime; }

// ====================================================
// DURUM
// ====================================================
let mangaState = 'idle';
let overlayElements = [];

console.log("🎬 Manga Translator content script yüklendi");

// ====================================================
// Z TUŞU
// ====================================================
document.addEventListener('keydown', (e) => {
    if (!isLatest()) return;
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target.isContentEditable) return;
    if (e.key.toLowerCase() !== 'z') return;

    console.log("⌨️ Z basıldı, durum:", mangaState);

    if (mangaState === 'overlay') { closeOverlay(); return; }
    if (mangaState === 'loading') return;

    mangaState = 'loading';
    chrome.runtime.sendMessage({ action: "z_translate" });
});

// ====================================================
// MESAJ DİNLEYİCİ
// ====================================================
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (!isLatest()) return;

    console.log("📨 Content mesaj:", request.action);

    // PING — background'un kontrol etmesi için
    if (request.action === "ping") {
        sendResponse({ pong: true });
        return;
    }

    if (request.action === "show_loading") {
        cleanupOverlay();
        mangaState = 'loading';
        showLoader();
    }
    else if (request.action === "show_result") {
        hideLoader();
        showOverlay(request.imageUrl);
        mangaState = 'overlay';
        console.log("🖼️ Overlay gösterildi");

        // İlk seferde Z ipucu
        chrome.storage.local.get(['z_hint_shown'], (s) => {
            if (!s.z_hint_shown) {
                showToast("💡 Artık Z tuşu ile çevir/kapat!", '#2d3436');
                chrome.storage.local.set({ z_hint_shown: true });
            }
        });
    }
    else if (request.action === "show_error") {
        hideLoader();
        mangaState = 'idle';
        showToast("❌ " + request.message, '#c0392b');
    }
    else if (request.action === "no_login") {
        hideLoader();
        mangaState = 'idle';
        showToast("⚠️ Önce extension'dan giriş yapın!", '#e67e22');
    }
});

// ====================================================
// LOADER
// ====================================================
function showLoader() {
    let loader = document.getElementById('manga-loader');
    if (!loader) {
        loader = document.createElement('div');
        loader.id = 'manga-loader';
        document.body.appendChild(loader);
    }
    loader.style.cssText = `
        position:fixed; top:20px; right:20px;
        background:#1a1a2e; color:#00ff88;
        padding:14px 24px; z-index:999999;
        border-radius:12px; font-family:sans-serif;
        font-weight:bold; font-size:14px;
        box-shadow:0 4px 20px rgba(0,0,0,0.5);
        display:flex; align-items:center; gap:10px;
    `;
    loader.innerHTML = `
        <span style="display:inline-block;width:16px;height:16px;
              border:3px solid rgba(0,255,136,0.3);border-top:3px solid #00ff88;
              border-radius:50%;animation:manga-spin .8s linear infinite"></span>
        Çevriliyor...
    `;

    if (!document.getElementById('manga-spinner-style')) {
        const s = document.createElement('style');
        s.id = 'manga-spinner-style';
        s.textContent = '@keyframes manga-spin{to{transform:rotate(360deg)}}';
        document.head.appendChild(s);
    }
}

function hideLoader() {
    const l = document.getElementById('manga-loader');
    if (l) l.style.display = 'none';
}

// ====================================================
// OVERLAY
// ====================================================
function showOverlay(url) {
    cleanupOverlay();

    const bg = document.createElement('div');
    bg.id = 'manga-overlay-bg';
    bg.style.cssText = `position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.4);z-index:999997;`;

    const img = document.createElement('img');
    img.id = 'manga-overlay-img';
    img.src = url;
    img.style.cssText = `position:fixed;top:0;left:0;width:100vw;height:100vh;object-fit:contain;z-index:999998;pointer-events:none;`;

    const closeBtn = document.createElement('button');
    closeBtn.id = 'manga-overlay-close';
    closeBtn.innerHTML = 'KAPAT <kbd style="background:#fff3;padding:2px 6px;border-radius:4px;margin-left:6px;font-size:11px">Z</kbd>';
    closeBtn.style.cssText = `
        position:fixed;top:16px;left:16px;z-index:999999;
        padding:10px 18px;background:#e74c3c;color:white;
        border:none;cursor:pointer;border-radius:8px;
        font-weight:bold;font-size:14px;font-family:sans-serif;
        box-shadow:0 2px 10px rgba(0,0,0,0.4);
        display:flex;align-items:center;
    `;
    closeBtn.onclick = () => closeOverlay();

    document.body.appendChild(bg);
    document.body.appendChild(img);
    document.body.appendChild(closeBtn);
    overlayElements = [bg, img, closeBtn];
}

function cleanupOverlay() {
    overlayElements.forEach(el => { if (el.parentNode) el.remove(); });
    overlayElements = [];
    ['manga-overlay-bg', 'manga-overlay-img', 'manga-overlay-close'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.remove();
    });
}

function closeOverlay() {
    cleanupOverlay();
    mangaState = 'idle';
    console.log("🚪 Overlay kapatıldı");
}

// ====================================================
// TOAST
// ====================================================
function showToast(msg, bgColor) {
    let t = document.getElementById('manga-toast');
    if (!t) { t = document.createElement('div'); t.id = 'manga-toast'; document.body.appendChild(t); }
    t.innerText = msg;
    t.style.cssText = `
        position:fixed;top:20px;right:20px;
        background:${bgColor || '#333'};color:white;
        padding:14px 24px;z-index:999999;
        border-radius:10px;font-family:sans-serif;
        font-weight:bold;font-size:14px;
        box-shadow:0 4px 20px rgba(0,0,0,0.4);
        opacity:1;transition:opacity .5s;display:block;
    `;
    setTimeout(() => {
        t.style.opacity = '0';
        setTimeout(() => { t.style.display = 'none'; t.style.opacity = '1'; }, 500);
    }, 3000);
}