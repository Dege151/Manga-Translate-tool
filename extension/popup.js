function logError(msg) {
    const d = document.getElementById('debugLog');
    if (d) { d.style.display = 'block'; d.innerText = "HATA: " + msg; }
    console.error(msg);
}

document.addEventListener('DOMContentLoaded', () => {
    try {
        const loginBtn = document.getElementById('loginBtn');
        const translateBtn = document.getElementById('translateBtn');
        const userInfoPanel = document.getElementById('userInfoPanel');
        const userNameSpan = document.getElementById('userName');
        const userCreditsDiv = document.getElementById('userCredits');
        const langSelect = document.getElementById('languageSelect');
        const guestMsg = document.getElementById('guestMsg');

        // Kayıtlı dili yükle
        chrome.storage.local.get(['target_lang'], (s) => {
            if (s.target_lang && langSelect) langSelect.value = s.target_lang;
        });

        // Dil değişince kaydet
        langSelect.addEventListener('change', () => {
            chrome.storage.local.set({ target_lang: langSelect.value });
        });

        // Otomatik giriş kontrolü
        chrome.identity.getAuthToken({ interactive: false }, (token) => {
            if (chrome.runtime.lastError || !token) showLoginUI();
            else handleLoginSuccess(token);
        });

        loginBtn.addEventListener('click', () => {
            chrome.identity.getAuthToken({ interactive: true }, (token) => {
                if (chrome.runtime.lastError) {
                    logError("Giriş Başarısız: " + chrome.runtime.lastError.message);
                    return;
                }
                handleLoginSuccess(token);
            });
        });

        function showLoginUI() {
            loginBtn.style.display = 'block';
            guestMsg.style.display = 'block';
            userInfoPanel.style.display = 'none';
            translateBtn.style.display = 'none';
            langSelect.style.display = 'none';
        }

        function showUserUI() {
            loginBtn.style.display = 'none';
            guestMsg.style.display = 'none';
            userInfoPanel.style.display = 'block';
            translateBtn.style.display = 'block';
            langSelect.style.display = 'block';
        }

        function handleLoginSuccess(token) {
            showUserUI();
            fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
                headers: { Authorization: `Bearer ${token}` }
            })
                .then(r => r.json())
                .then(user => {
                    userNameSpan.innerText = user.given_name || "Kullanıcı";
                    fetchCredits(token);

                    translateBtn.onclick = () => {
                        const lang = langSelect.value;
                        chrome.storage.local.set({ target_lang: lang });

                        chrome.runtime.sendMessage({
                            action: "start_translation",
                            token: token,
                            email: user.email,
                            target_lang: lang
                        });

                        // Mesaj gönderildikten sonra kapat
                        setTimeout(() => window.close(), 100);
                    };
                })
                .catch(err => logError("Profil hatası: " + err.message));
        }

        function fetchCredits(token) {
            userCreditsDiv.innerText = "Kredi yükleniyor...";
            fetch('http://127.0.0.1:5000/me', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token: token })
            })
                .then(r => { if (!r.ok) throw new Error("Bağlantı hatası"); return r.json(); })
                .then(data => {
                    if (data.is_premium) {
                        userCreditsDiv.innerText = "∞ Limitsiz (Premium)";
                        userCreditsDiv.style.color = "green";
                    } else {
                        userCreditsDiv.innerText = `${data.credits} Kredi Kaldı`;
                        if (data.credits <= 0) userCreditsDiv.style.color = "red";
                    }
                })
                .catch(err => { userCreditsDiv.innerText = "Sunucu Hatası"; });
        }

    } catch (e) { logError("Genel Hata: " + e.message); }
});