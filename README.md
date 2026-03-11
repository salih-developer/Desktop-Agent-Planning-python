# Desktop Agent

Yerel olarak çalışan, Ollama tabanlı bir masaüstü AI asistanı. Dosya okuma/yazma, web arama, komut çalıştırma ve uzun süreli hafıza özellikleriyle donatılmış, araç çağrısı yapabilen bir ReAct döngüsü üzerine inşa edilmiştir.

---

## Özellikler

### ReAct Döngüsü
- **Gözlemle → Düşün → Davran** döngüsüyle adım adım görev çözer
- Ollama native tool calling desteği
- Text-format tool call fallback: `<function=...>` biçiminde yanıt üreten modeller de çalışır
- Configurable maksimum iterasyon sayısı (varsayılan: 15)
- Her oturum için `logs/` klasörüne detaylı `.log` trace dosyası kaydeder

### Araçlar (Tools)
| Araç | Açıklama |
|------|----------|
| `file_read` | Dosya okur; büyük dosyalar için satır aralığı desteği |
| `file_write` | Yeni dosya oluşturur veya üzerine yazar |
| `file_edit` | Dosyada belirli metin bloğunu yerinde değiştirir |
| `shell_run` | Windows cmd.exe komutu çalıştırır; timeout desteği |
| `web_search` | DuckDuckGo üzerinden web araması yapar |
| `web_fetch` | URL içeriğini çeker (HTML → metin dönüşümü) |
| `dir_tree` | Dizin ağacı çıktısı verir |
| `find_files` | Glob pattern ile dosya arar |

### Semantik Hafıza
- Her konuşma `sqlite-vec` vektör veritabanında saklanır
- `nomic-embed-text` modeli ile 768-boyutlu embedding üretilir
- Yeni sorguda en ilgili geçmiş etkileşimler KNN ile bulunur ve bağlam olarak eklenir
- Hafıza veritabanı: `~/.desktop_agent/memory.db`

### Güvenlik
- **Prompt Injection Koruması**: Araç çıktılarında gömülü talimat kalıpları tespit edilir (`ignore previous`, `act as`, `you are` vb.) — `[⚠ INJECTION WARNING]` ile işaretlenir ve LLM uyarılır
- UI'da enjeksiyon uyarıları turuncu renkte vurgulanır
- **Engellenen komutlar**: `rm -rf /`, `format`, `dd if=` gibi yıkıcı komutlar çalıştırılmaz
- `prompt_injection_protection` config ile açılıp kapatılabilir

### Dosya ve Görsel Eki
- Sohbet kutusuna **dosya** ve **klasör** eklenebilir (📎 butonu)
- PNG, JPG, PDF, TXT, MD, CS, SQL, JSON, YAML gibi formatlar desteklenir
- Ekler agent'a bağlam olarak aktarılır

### Arayüz (UI)
- **CustomTkinter** tabanlı masaüstü uygulaması
- Araç çağrıları sohbet akışı içinde inline olarak gösterilir (`⚙ araç çağrıları` bloğu)
- Kullanıcı mesajı → araç çağrıları → agent yanıtı sırayla akar
- Görev takip paneli: her adımın durumu ve geçen süre
- **Stop** butonu ile çalışan agent iptal edilebilir
- Ayarlar iletişim kutusu: model, workspace, timeout, iterasyon limiti, hafıza boyutu ve güvenlik ayarları

### Yapılandırma
Ayarlar `~/.desktop_agent/config.yaml` dosyasında saklanır:

```yaml
ollama_base_url: http://localhost:11434
react_model: qwen2.5:7b
react_max_iterations: 15
workspace: C:\source
max_conversation_history: 8
max_tool_output_chars: 8000
prompt_injection_protection: true
use_react_loop: true
embedding_model: nomic-embed-text
memory_top_k: 5
shell_timeout_seconds: 30
```

---

## Kurulum

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
python main.py
```

> Ollama'nın `http://localhost:11434` adresinde çalışıyor olması gerekir.

---

## Proje Yapısı

```
DesktopAgentPlanning/
├── main.py                  # Giriş noktası
├── config.py                # AppConfig dataclass + YAML yükleme
├── logger.py                # Loglama
├── core/
│   ├── agent.py             # Orkestratör
│   ├── react_loop.py        # ReAct döngüsü
│   ├── react_tracer.py      # Oturum trace yazıcısı
│   ├── planner.py           # Görev planlayıcı (Pydantic)
│   ├── executor.py          # Paralel görev çalıştırıcı
│   └── synthesizer.py       # Streaming final yanıt
├── memory/
│   ├── store.py             # sqlite-vec vektör deposu
│   └── embedder.py          # Ollama embedding istemcisi
├── tools/
│   ├── registry.py          # Araç kaydı ve fabrika
│   ├── file_tools.py        # file_read / file_write / file_edit
│   ├── shell_tool.py        # shell_run
│   ├── web_tools.py         # web_search / web_fetch
│   └── filesystem_tool.py   # dir_tree / find_files
├── ui/
│   ├── app.py               # CTk root, threading köprüsü, SettingsDialog
│   └── panels/
│       ├── chat_panel.py    # Sohbet + ek yönetimi
│       └── task_panel.py    # Görev durum listesi
│   └── widgets/
│       ├── message_bubble.py
│       ├── task_item.py
│       └── tool_output_block.py
└── logs/                    # Oturum trace dosyaları (.log)
```
