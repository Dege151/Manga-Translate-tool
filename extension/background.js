// ====================================================
// EXTENSION YÜKLENDİĞİNDE: Tüm açık tab'lara content script enjekte et
// ====================================================
chrome.runtime.onInstalled.addListener(() => {
    console.log("🔧 Extension yüklendi/güncellendi — tab'lara enjeksiyon yapılıyor");
    chrome.tabs.query({}, (tabs) => {
        tabs.forEach(tab => {
            if (tab.url && !tab.url.startsWith('chrome://') && !tab.url.startsWith('chrome-extension://')) {
                chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    files: ['content.js']
                }).catch(() => { });
            }
        });
    });
});

// ====================================================
// CONTENT SCRIPT AKTİF Mİ KONTROL ET, DEĞİLSE ENJEKTE ET
// ====================================================
async function ensureContentScript(tabId) {
    try {
        const response = await chrome.tabs.sendMessage(tabId, { action: "ping" });
        if (response && response.pong) {
            console.log("✅ Content script zaten aktif, tab:", tabId);
            return;
        }
        throw new Error("Geçersiz yanıt");
    } catch (e) {
        console.log("⚠️ Content script yok, enjekte ediliyor, tab:", tabId);
        await chrome.scripting.executeScript({
            target: { tabId: tabId },
            files: ['content.js']
        });
        await new Promise(r => setTimeout(r, 300));
        console.log("✅ Content script enjekte edildi, tab:", tabId);
    }
}

// ====================================================
// ANA ÇEVİRİ FONKSİYONU (ASYNC)
// ====================================================
async function doTranslation(token, targetLang, tabId, windowId) {
    try {
        // 1) SCREENSHOT (temiz ekran — loader yok)
        console.log("📸 Screenshot alınıyor...");
        const dataUrl = await chrome.tabs.captureVisibleTab(windowId, { format: "jpeg", quality: 80 });
        if (!dataUrl) throw new Error("Screenshot boş döndü");
        console.log("📸 Screenshot alındı, boyut:", dataUrl.length);

        // 2) Content script hazır mı?
        await ensureContentScript(tabId);

        // 3) LOADER GÖSTER (screenshot SONRASI)
        chrome.tabs.sendMessage(tabId, { action: "show_loading" });
        console.log("⏳ Loader gösterildi");

        // 4) API'ye GÖNDER
        console.log("🌐 Sunucuya gönderiliyor...");
        const res = await fetch('http://127.0.0.1:5000/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: dataUrl,
                token: token,
                target_language: targetLang
            })
        });

        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Sunucu Hatası (${res.status}): ${errText}`);
        }

        const data = await res.json();
        console.log("✅ Sunucu yanıtı:", Object.keys(data));

        // 5) SONUCU SAYFAYA GÖNDER
        if (data.processed_image) {
            chrome.tabs.sendMessage(tabId, {
                action: "show_result",
                imageUrl: data.processed_image
            });
            console.log("🖼️ Overlay gönderildi, kalan kredi:", data.remaining_credits);
        } else {
            throw new Error("Sunucu processed_image döndürmedi");
        }

    } catch (err) {
        console.error("❌ Çeviri hatası:", err.message);
        try {
            await ensureContentScript(tabId);
            chrome.tabs.sendMessage(tabId, {
                action: "show_error",
                message: err.message
            });
        } catch (e2) {
            console.error("❌ Hata mesajı da gönderilemedi:", e2);
        }
    }
}

// ====================================================
// MESAJ DİNLEYİCİ
// ====================================================
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

    // --- POPUP'TAN ÇEVİRİ ---
    if (request.action === "start_translation") {
        console.log("📬 Popup → çeviri isteği, dil:", request.target_lang);
        chrome.storage.local.set({ target_lang: request.target_lang });

        chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
            const tab = tabs[0];
            if (!tab) { console.error("❌ Aktif tab yok"); return; }
            console.log("📋 Tab:", tab.id, tab.url);
            await doTranslation(request.token, request.target_lang, tab.id, tab.windowId);
        });
        return true;
    }

    // --- Z TUŞU ÇEVİRİ ---
    if (request.action === "z_translate") {
        console.log("⌨️ Z tuşu → çeviri isteği, tab:", sender.tab?.id);
        const tabId = sender.tab.id;
        const windowId = sender.tab.windowId;

        (async () => {
            try {
                const stored = await chrome.storage.local.get(['target_lang']);
                const targetLang = stored.target_lang || 'Turkish';
                console.log("🌍 Kayıtlı dil:", targetLang);

                const token = await new Promise((resolve, reject) => {
                    chrome.identity.getAuthToken({ interactive: false }, (t) => {
                        if (chrome.runtime.lastError || !t)
                            reject(new Error("Giriş yapılmamış"));
                        else resolve(t);
                    });
                });
                console.log("🔑 Token alındı");
                await doTranslation(token, targetLang, tabId, windowId);
            } catch (e) {
                console.error("❌ Z hatası:", e.message);
                chrome.tabs.sendMessage(tabId, { action: "no_login" });
            }
        })();
        return true;
    }
});