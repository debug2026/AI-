
import React, { useState, useEffect, useRef, useCallback, Component, ErrorInfo } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type, Modality } from "@google/genai";

// --- API Key & Initialization ---

const getApiKey = () => {
  try {
    // @ts-ignore
    if (typeof process !== 'undefined' && process.env && process.env.API_KEY) {
      // @ts-ignore
      return process.env.API_KEY;
    }
  } catch (e) {
    console.warn("Error accessing process.env", e);
  }
  return "";
};

const API_KEY = getApiKey();
let ai: GoogleGenAI | null = null;
try {
  ai = new GoogleGenAI({ apiKey: API_KEY || "dummy-key" });
} catch (e) {
  console.error("Failed to initialize GoogleGenAI", e);
}

// --- Error Boundary ---

class ErrorBoundary extends Component<{ children: React.ReactNode }, { hasError: boolean, error: Error | null }> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-screen bg-gray-900 flex flex-col items-center justify-center text-white p-8">
          <h1 className="text-3xl font-bold text-red-500 mb-4">应用运行出错</h1>
          <p className="text-gray-300 mb-4 text-center max-w-lg">检测到致命错误，请检查 API Key 配置或刷新页面重试。</p>
          <pre className="bg-black/50 p-4 rounded text-xs text-red-300 overflow-auto max-w-2xl w-full border border-red-900">
            {this.state.error?.toString()}
          </pre>
          <button onClick={() => window.location.reload()} className="mt-6 px-6 py-2 bg-purple-600 rounded hover:bg-purple-500">刷新页面</button>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- IndexedDB Utilities ---

const DB_NAME = 'AI_Storyboard_DB_V3';
const DB_VERSION = 1;
const STORE_DATA = 'data';
const STORE_SCENES = 'scenes';

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) {
      reject(new Error("IndexedDB not supported"));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (e) => {
      const db = (e.target as any).result;
      if (!db.objectStoreNames.contains(STORE_DATA)) db.createObjectStore(STORE_DATA, { keyPath: 'key' });
      if (!db.objectStoreNames.contains(STORE_SCENES)) db.createObjectStore(STORE_SCENES, { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// Get from key-value store (STORE_DATA)
async function dbGetData(key: string): Promise<any> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_DATA, 'readonly');
      const store = tx.objectStore(STORE_DATA);
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result ? request.result.value : null);
      request.onerror = () => reject(request.error);
    });
  } catch (e) {
    return null;
  }
}

// Put into key-value store
async function dbPutData(key: string, value: any): Promise<void> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_DATA, 'readwrite');
      const store = tx.objectStore(STORE_DATA);
      const request = store.put({ key, value });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (e) {}
}

// Get all scenes
async function dbGetAllScenes(): Promise<any[]> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_SCENES, 'readonly');
      const store = tx.objectStore(STORE_SCENES);
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error);
    });
  } catch (e) {
    return [];
  }
}

// Put scene
async function dbPutScene(scene: any): Promise<void> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_SCENES, 'readwrite');
      const store = tx.objectStore(STORE_SCENES);
      const request = store.put(scene);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (e) {}
}

async function dbClearStore(storeName: string): Promise<void> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const request = store.clear();
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (e) {}
}

// --- Utilities ---

function cleanJson(text: string) {
  try {
    if (!text) return {};
    // Aggressively clean markdown
    let cleaned = text.replace(/```json/g, '').replace(/```/g, '');
    const firstBrace = cleaned.indexOf('{');
    const lastBrace = cleaned.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace !== -1) {
      cleaned = cleaned.substring(firstBrace, lastBrace + 1);
    }
    return JSON.parse(cleaned);
  } catch (e) {
    console.error("JSON Parse Error:", e, text);
    throw new Error("AI 响应格式无法解析");
  }
}

function decodeBase64(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(data: Uint8Array, ctx: AudioContext): Promise<AudioBuffer> {
  const bufferCopy = data.buffer.slice(0);
  return await ctx.decodeAudioData(bufferCopy);
}

// --- Task Queue ---

class RequestQueue {
  private queue: (() => Promise<void>)[] = [];
  private activeCount = 0;
  private maxConcurrent = 3;

  add(task: () => Promise<void>) {
    this.queue.push(task);
    this.process();
  }

  private process() {
    if (this.activeCount >= this.maxConcurrent || this.queue.length === 0) return;
    const task = this.queue.shift();
    if (!task) return;

    this.activeCount++;
    task().finally(() => {
      this.activeCount--;
      this.process();
    });
  }
}

const imageGenQueue = new RequestQueue();

// --- Types ---

interface Character {
  id: string;
  name: string;
  description: string;
}

interface StoryboardScene {
  id: number;
  originalText: string;
  visualDescription: string;
  images: string[];
  selectedImageIndex: number;
  isGeneratingImages: boolean;
  audioData?: ArrayBuffer;
  audioDuration?: number; 
}

type ImageModelType = 'gemini' | 'comfyui';
type TTSProviderType = 'gemini' | 'browser';

interface AppSettings {
  style: string;
  aspectRatio: string;
  llmModel: string;
  imageModel: ImageModelType;
  comfyUiApiUrl: string;
  ttsProvider: TTSProviderType;
  ttsVoice: string;
  ttsRate: number;
  ttsPitch: number;
}

// --- Constants ---

const STYLES = [
  "Cinematic (电影感)",
  "Anime (日式动漫)",
  "3D Render (3D渲染)",
  "Cyberpunk (赛博朋克)",
  "Watercolor (水彩)",
  "Sketch (素描草图)",
  "Photorealistic (写实摄影)"
];

const ASPECT_RATIOS = [
  { label: "16:9 (横屏)", value: "16:9", width: 1280, height: 720 },
  { label: "9:16 (竖屏)", value: "9:16", width: 720, height: 1280 },
  { label: "1:1 (方图)", value: "1:1", width: 1024, height: 1024 },
  { label: "2.35:1 (宽银幕)", value: "2.35:1", width: 1920, height: 817 },
];

const DEFAULT_SETTINGS: AppSettings = {
  style: "Cinematic (电影感)",
  aspectRatio: "16:9",
  llmModel: "gemini-2.5-flash",
  imageModel: "gemini",
  comfyUiApiUrl: "http://127.0.0.1:8188",
  ttsProvider: "gemini",
  ttsVoice: "Kore",
  ttsRate: 1.0,
  ttsPitch: 0,
};

// --- Components ---

const SettingsModal = ({
  isOpen, onClose, settings, onSave
}: {
  isOpen: boolean; onClose: () => void; settings: AppSettings; onSave: (s: AppSettings) => void;
}) => {
  const [localSettings, setLocalSettings] = useState(settings);

  useEffect(() => {
    if (isOpen) setLocalSettings(settings);
  }, [isOpen, settings]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl">
        <div className="flex items-center justify-between p-6 border-b border-gray-800">
          <h2 className="text-xl font-bold text-white">⚙️ 全局设置</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">&times;</button>
        </div>
        <div className="p-6 space-y-8">
          <section>
            <h3 className="text-purple-400 text-sm font-bold uppercase tracking-wider mb-4 border-b border-gray-800 pb-2">模型配置</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-gray-400 text-xs mb-2">默认 LLM</label>
                <select className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white" value={localSettings.llmModel} onChange={e => setLocalSettings({...localSettings, llmModel: e.target.value})}>
                  <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                  <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
                </select>
              </div>
              <div>
                <label className="block text-gray-400 text-xs mb-2">生图模型</label>
                <select className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white" value={localSettings.imageModel} onChange={e => setLocalSettings({...localSettings, imageModel: e.target.value as ImageModelType})}>
                  <option value="gemini">Gemini 2.5 Flash Image</option>
                  <option value="comfyui">ComfyUI (需本地启动)</option>
                </select>
              </div>
            </div>
          </section>
          <section>
            <h3 className="text-green-400 text-sm font-bold uppercase tracking-wider mb-4 border-b border-gray-800 pb-2">语音配置</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
              <div>
                <label className="block text-gray-400 text-xs mb-2">TTS 提供商</label>
                <select className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white" value={localSettings.ttsProvider} onChange={e => setLocalSettings({...localSettings, ttsProvider: e.target.value as TTSProviderType})}>
                  <option value="gemini">Gemini TTS (推荐)</option>
                  <option value="browser">浏览器原生</option>
                </select>
              </div>
              <div>
                <label className="block text-gray-400 text-xs mb-2">音色</label>
                <select className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white" value={localSettings.ttsVoice} onChange={e => setLocalSettings({...localSettings, ttsVoice: e.target.value})}>
                  <option value="Kore">Kore</option>
                  <option value="Puck">Puck</option>
                  <option value="Charon">Charon</option>
                  <option value="Fenrir">Fenrir</option>
                  <option value="Aoede">Aoede</option>
                </select>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-gray-400 text-xs mb-2 flex justify-between"><span>语速</span><span className="text-white">{localSettings.ttsRate}x</span></label>
                <input type="range" min="0.5" max="2.0" step="0.1" className="w-full accent-green-500" value={localSettings.ttsRate} onChange={e => setLocalSettings({...localSettings, ttsRate: parseFloat(e.target.value)})} />
              </div>
              <div>
                <label className="block text-gray-400 text-xs mb-2 flex justify-between"><span>语调</span><span className="text-white">{localSettings.ttsPitch}</span></label>
                <input type="range" min="-1200" max="1200" step="100" className="w-full accent-green-500" value={localSettings.ttsPitch} onChange={e => setLocalSettings({...localSettings, ttsPitch: parseInt(e.target.value)})} />
              </div>
            </div>
          </section>
        </div>
        <div className="p-6 border-t border-gray-800 flex justify-end gap-3">
          <button onClick={onClose} className="px-4 py-2 rounded text-sm text-gray-400 hover:text-white">取消</button>
          <button onClick={() => { onSave(localSettings); onClose(); }} className="px-6 py-2 rounded text-sm bg-purple-600 hover:bg-purple-500 text-white font-bold">保存</button>
        </div>
      </div>
    </div>
  );
};

const Header = ({ 
  onReset, settings, setSettings, isExporting, onExport, onOpenSettings
}: { 
  onReset: () => void, settings: AppSettings, setSettings: (s: AppSettings) => void,
  isExporting: boolean, onExport: () => void, onOpenSettings: () => void
}) => (
  <header className="bg-gray-900 border-b border-gray-800 p-4 flex items-center justify-between sticky top-0 z-20 shadow-md shrink-0 h-16">
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-blue-600 rounded-lg flex items-center justify-center text-white font-bold shadow-lg">AI</div>
        <h1 className="text-lg font-bold text-white tracking-wide hidden lg:block">AI Storyboard</h1>
      </div>
      <div className="h-6 w-px bg-gray-700 mx-2 hidden md:block"></div>
      <div className="flex items-center gap-2">
        <select value={settings.style} onChange={(e) => setSettings({...settings, style: e.target.value})} className="bg-gray-800 text-xs text-white border border-gray-700 rounded px-2 py-1.5 focus:border-purple-500 outline-none">
          {STYLES.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <select value={settings.aspectRatio} onChange={(e) => setSettings({...settings, aspectRatio: e.target.value})} className="bg-gray-800 text-xs text-white border border-gray-700 rounded px-2 py-1.5 focus:border-purple-500 outline-none">
          {ASPECT_RATIOS.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
        </select>
      </div>
    </div>
    <div className="flex items-center gap-3">
       <button onClick={onOpenSettings} className="text-gray-400 hover:text-white p-2 rounded hover:bg-gray-800" title="设置">
         <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
       </button>
       <button onClick={onExport} disabled={isExporting} className={`text-xs px-4 py-2 rounded font-bold text-white transition-all flex items-center gap-2 ${isExporting ? 'bg-gray-700 cursor-wait' : 'bg-green-600 hover:bg-green-500 shadow-lg'}`}>
        {isExporting ? '合成中...' : '🎬 导出视频'}
      </button>
      <button onClick={onReset} className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white px-3 py-2 rounded border border-gray-700">清空</button>
    </div>
  </header>
);

const SceneCard = ({ 
  scene, index, onSelectImage, onRegenerateImages, onDragStart, onDragOver, onDrop, onOpenLightbox
}: {
  scene: StoryboardScene, index: number,
  onSelectImage: (sid: number, i: number) => void,
  onRegenerateImages: (s: StoryboardScene) => void,
  onDragStart: (i: number) => void, onDragOver: (e: React.DragEvent, i: number) => void, onDrop: (i: number) => void,
  onOpenLightbox: (url: string) => void
}) => {
  return (
    <div 
      className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden shadow-sm hover:shadow-md transition-all group shrink-0 flex flex-col md:flex-row h-auto md:h-96" 
      draggable
      onDragStart={() => onDragStart(index)}
      onDragOver={(e) => onDragOver(e, index)}
      onDrop={() => onDrop(index)}
    >
      <div className="p-4 border-b md:border-b-0 md:border-r border-gray-700 flex flex-col justify-between w-full md:w-1/3 bg-gray-800/50">
        <div className="flex-1 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-gray-700">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-gray-500 cursor-grab active:cursor-grabbing hover:text-gray-300">
              <span className="font-mono text-lg font-bold opacity-50">#{index + 1}</span>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 8h16M4 16h16"></path></svg>
            </div>
            {scene.audioDuration && (
               <span className="text-[10px] bg-green-900/30 text-green-400 px-2 py-1 rounded-full border border-green-900/50 flex items-center gap-1 shrink-0">
                 TTS: {scene.audioDuration.toFixed(1)}s
               </span>
             )}
          </div>
          <p className="text-white text-sm font-medium leading-relaxed mb-3 select-text whitespace-pre-wrap">"{scene.originalText}"</p>
          <div className="bg-gray-900/50 p-2 rounded text-xs text-gray-500 border border-gray-700/50">
             <span className="text-purple-400 font-semibold block mb-1">画面描述:</span> 
             <p className="italic select-text">{scene.visualDescription}</p>
          </div>
        </div>
        <button onClick={() => onRegenerateImages(scene)} disabled={scene.isGeneratingImages} className="mt-3 w-full text-xs bg-gray-700 hover:bg-gray-600 text-white px-3 py-3 rounded border border-gray-600 flex items-center justify-center gap-2 transition-colors shrink-0">
             {scene.isGeneratingImages ? <span className="animate-spin">⟳</span> : <span>↻ 重新绘图</span>}
        </button>
      </div>

      <div className="p-3 bg-gray-900/30 flex-1 flex flex-col justify-center overflow-hidden">
         {scene.isGeneratingImages && scene.images.length === 0 ? (
           <div className="h-full w-full flex flex-col items-center justify-center text-purple-400 animate-pulse bg-gray-800/50 rounded-lg">
             <span className="text-sm">正在绘制变体 (队列中)...</span>
           </div>
         ) : (
           <div className="grid grid-cols-2 gap-3 h-full">
              {scene.images.map((img, idx) => (
                <div key={idx} className={`relative rounded-lg overflow-hidden cursor-pointer border-2 transition-all group/img h-full bg-black ${scene.selectedImageIndex === idx ? 'border-green-500 shadow-lg' : 'border-transparent hover:border-gray-500'}`} onClick={() => onSelectImage(scene.id, idx)} onDoubleClick={() => onOpenLightbox(img)}>
                  <img src={img} className="w-full h-full object-contain" loading="lazy" />
                  {scene.selectedImageIndex === idx && <div className="absolute top-2 right-2 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center text-black font-bold text-[10px] shadow-sm z-10">✓</div>}
                </div>
              ))}
              {[...Array(Math.max(0, 4 - scene.images.length))].map((_, i) => (
                 <div key={`empty-${i}`} className="h-full bg-gray-800 rounded-lg flex items-center justify-center border border-gray-700 border-dashed"><span className="text-gray-600 text-xs">Waiting...</span></div>
              ))}
           </div>
         )}
      </div>
    </div>
  );
}

const CharacterEdit = ({ 
  char, onUpdate, onDelete 
}: { 
  char: Character, onUpdate: (c: Character) => void, onDelete: (id: string) => void 
}) => (
  <div className="bg-gray-800/80 rounded-lg p-3 border border-gray-700 mb-3 group">
    <div className="flex justify-between items-center mb-2">
      <input 
        className="bg-transparent text-purple-400 font-bold text-sm outline-none w-2/3"
        value={char.name}
        onChange={e => onUpdate({...char, name: e.target.value})}
        placeholder="角色名"
      />
      <button onClick={() => onDelete(char.id)} className="text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity">×</button>
    </div>
    <textarea 
      className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-xs text-white placeholder-gray-600 outline-none resize-none h-16"
      value={char.description}
      onChange={e => onUpdate({...char, description: e.target.value})}
      placeholder="输入外貌描述，将自动注入分镜..."
    />
  </div>
);

// --- App Container ---

const AppContent = () => {
  const [script, setScript] = useState("");
  const [characters, setCharacters] = useState<Character[]>([]);
  const [scenes, setScenes] = useState<StoryboardScene[]>([]);
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState("");
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);

  // Load Data
  useEffect(() => {
    const loadData = async () => {
      try {
        const savedScript = await dbGetData('script');
        if (savedScript) setScript(savedScript);
        
        const savedChars = await dbGetData('characters');
        if (Array.isArray(savedChars)) setCharacters(savedChars);

        const savedSettings = await dbGetData('settings');
        if (savedSettings) setSettings({ ...DEFAULT_SETTINGS, ...savedSettings });
        
        const savedScenes = await dbGetAllScenes();
        if (Array.isArray(savedScenes)) {
           setScenes(savedScenes.sort((a,b) => a.id - b.id));
        }
      } catch (e) {
        console.error("DB Load failed", e);
      }
    };
    loadData();
  }, []);

  useEffect(() => { dbPutData('script', script); }, [script]);
  useEffect(() => { dbPutData('characters', characters); }, [characters]);
  useEffect(() => { dbPutData('settings', settings); }, [settings]);
  
  useEffect(() => {
    // Only save scenes when they change
    const saveAll = async () => {
       for (const s of scenes) await dbPutScene(s);
    };
    if (scenes.length > 0) saveAll();
  }, [scenes]);

  const audioContextRef = useRef<AudioContext | null>(null);
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      try {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      } catch (e) {
        console.error("AudioContext not supported");
      }
    }
    return audioContextRef.current;
  }, []);

  const handleReset = async () => {
    if (confirm("清空当前项目？此操作不可撤销。")) {
      setScript("");
      setScenes([]);
      setCharacters([]);
      await dbClearStore(STORE_SCENES);
      await dbPutData('script', "");
      await dbPutData('characters', []);
    }
  };

  const handleAnalyze = async () => {
    if (!ai) { alert("AI 未初始化，请检查 API Key"); return; }
    if (!script.trim()) return;
    setIsAnalyzing(true);
    setScenes([]); 
    await dbClearStore(STORE_SCENES); 

    try {
      const response = await ai.models.generateContent({
        model: settings.llmModel,
        contents: script,
        config: {
          systemInstruction: `你是分镜师。分析剧本：1.提取主要角色(name, description)。2.拆解为分镜(scene)。3.为每个分镜生成 visualDescription。返回纯净JSON。`,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              characters: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: { id: { type: Type.STRING }, name: { type: Type.STRING }, description: { type: Type.STRING } },
                  required: ["id", "name", "description"]
                }
              },
              scenes: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: { id: { type: Type.INTEGER }, originalText: { type: Type.STRING }, visualDescription: { type: Type.STRING } },
                  required: ["id", "originalText", "visualDescription"]
                }
              }
            },
            required: ["characters", "scenes"]
          }
        },
      });

      const result = cleanJson(response.text || "{}");
      const newChars = result.characters || [];
      const parsedScenes = result.scenes || [];
      
      setCharacters(newChars);

      const newScenes: StoryboardScene[] = parsedScenes.map((s: any, i: number) => ({
        id: Date.now() + i,
        originalText: s.originalText,
        visualDescription: s.visualDescription,
        images: [],
        selectedImageIndex: 0,
        isGeneratingImages: false,
      }));

      setScenes(newScenes);
      // Trigger generation with new characters explicitly passed
      newScenes.forEach(s => triggerImageGeneration(s.id, s.visualDescription, newChars));

    } catch (error) {
      console.error(error);
      alert("分析失败：" + (error as any).message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const triggerImageGeneration = (sceneId: number, prompt: string, currentChars: Character[]) => {
    if (!ai) return;
    let injection = "";
    const lowerPrompt = prompt.toLowerCase();
    currentChars.forEach(char => {
      if (lowerPrompt.includes(char.name.toLowerCase())) {
        injection += `(${char.name}, description: ${char.description}), `;
      }
    });
    const enrichedPrompt = `${settings.style} style. ${injection} ${prompt}`;
    
    imageGenQueue.add(async () => {
      setScenes(prev => prev.map(s => s.id === sceneId ? { ...s, isGeneratingImages: true } : s));
      try {
        let newImages: string[] = [];
        if (settings.imageModel === 'gemini') {
           const requests = Array(4).fill(0).map(() => 
             ai!.models.generateContent({
               model: 'gemini-2.5-flash-image',
               contents: { parts: [{ text: enrichedPrompt }] },
               config: { 
                 // @ts-ignore
                 imageConfig: { aspectRatio: settings.aspectRatio } 
               }
             })
           );
           const responses = await Promise.all(requests);
           responses.forEach(res => {
             res.candidates?.[0]?.content?.parts?.forEach(part => {
               if (part.inlineData) newImages.push(`data:${part.inlineData.mimeType};base64,${part.inlineData.data}`);
             });
           });
        } else {
           newImages = Array(4).fill(0).map((_,i) => `https://via.placeholder.com/1280x720/333/ccc?text=ComfyUI+Stub+${i}`);
        }
        
        setScenes(prev => prev.map(s => 
          s.id === sceneId ? { ...s, images: newImages, isGeneratingImages: false, selectedImageIndex: 0 } : s
        ));
      } catch (e) {
        setScenes(prev => prev.map(s => s.id === sceneId ? { ...s, isGeneratingImages: false } : s));
      }
    });
  };

  const handleRegenerateImages = (scene: StoryboardScene) => {
    setScenes(prev => prev.map(s => s.id === scene.id ? { ...s, images: [] } : s));
    triggerImageGeneration(scene.id, scene.visualDescription, characters);
  };

  const handleExportVideo = async () => {
    if (scenes.length === 0) return;
    const ctx = getAudioContext();
    if (!ctx) { alert("您的浏览器不支持音频处理，无法导出。"); return; }
    
    setIsExporting(true);
    setExportProgress("语音生成中...");

    try {
      if (ctx.state === 'suspended') await ctx.resume();
      const updatedScenes = [...scenes];
      
      await Promise.all(updatedScenes.map(async (scene, index) => {
        if (scene.audioData) return;
        if (settings.ttsProvider === 'gemini' && ai) {
            try {
              const response = await ai.models.generateContent({
                model: "gemini-2.5-flash-preview-tts",
                contents: [{ parts: [{ text: scene.originalText || " " }] }],
                config: {
                  responseModalities: [Modality.AUDIO],
                  speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: settings.ttsVoice } } },
                },
              });
              const base64 = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
              if (base64) {
                const rawBytes = decodeBase64(base64);
                const audioBuffer = await decodeAudioData(rawBytes, ctx);
                updatedScenes[index] = { ...scene, audioData: rawBytes.buffer, audioDuration: audioBuffer.duration };
              }
            } catch(e) {}
        } else {
             // Fallback duration calc
             updatedScenes[index] = { ...scene, audioDuration: Math.max(2, (scene.originalText.length || 1) * 0.25) };
        }
      }));
      setScenes(updatedScenes);

      setExportProgress("视频渲染中...");
      await recordVideo(updatedScenes, ctx);
    } catch (e) {
      alert("导出错误: " + (e as any).message);
    } finally {
      setIsExporting(false);
      setExportProgress("");
    }
  };

  const recordVideo = async (scenesToRecord: StoryboardScene[], audioCtx: AudioContext) => {
     if (typeof MediaRecorder === 'undefined') { throw new Error("此浏览器不支持 MediaRecorder"); }
     
     const ratioConfig = ASPECT_RATIOS.find(r => r.value === settings.aspectRatio) || ASPECT_RATIOS[0];
     const width = ratioConfig.width;
     const height = ratioConfig.height;
     const canvas = document.createElement('canvas');
     canvas.width = width;
     canvas.height = height;
     const ctx = canvas.getContext('2d');
     if (!ctx) return;
     
     const dest = audioCtx.createMediaStreamDestination();
     const canvasStream = canvas.captureStream(30); 
     const combinedStream = new MediaStream([...canvasStream.getVideoTracks(), ...dest.stream.getAudioTracks()]);
     
     const mimeType = ['video/mp4', 'video/webm;codecs=h264', 'video/webm'].find(t => MediaRecorder.isTypeSupported(t)) || '';
     if (!mimeType) throw new Error("无支持的视频编码格式");

     const recorder = new MediaRecorder(combinedStream, { mimeType, videoBitsPerSecond: 8000000 });
     const chunks: Blob[] = [];
     recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
     recorder.start();

     const loadedImages = await Promise.all(scenesToRecord.map(async (scene) => {
      if (scene.images?.length > 0) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = scene.images[scene.selectedImageIndex];
        await new Promise(r => { img.onload = r; img.onerror = r; });
        return img;
      }
      return null;
    }));

    for (let i = 0; i < scenesToRecord.length; i++) {
      const scene = scenesToRecord[i];
      const img = loadedImages[i];
      const nextImg = (i < scenesToRecord.length - 1) ? loadedImages[i+1] : null;

      setExportProgress(`录制分镜 ${i + 1}/${scenesToRecord.length}`);

      let effectiveDuration = 2000;
      const rate = Math.max(0.1, Math.min(settings.ttsRate, 4.0));

      if (scene.audioData) {
        const buffer = await decodeAudioData(new Uint8Array(scene.audioData), audioCtx);
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.playbackRate.value = rate;
        source.detune.value = settings.ttsPitch; 
        source.connect(dest);
        source.start();
        effectiveDuration = (buffer.duration / rate) * 1000;
      } else {
        effectiveDuration = (scene.audioDuration || 2) * 1000 / rate;
      }
      
      const startTime = Date.now();
      while (true) {
        const now = Date.now();
        const elapsed = now - startTime;
        if (elapsed >= effectiveDuration) break;
        const progress = elapsed / effectiveDuration;
        
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, width, height);

        ctx.save();
        const scale = 1.0 + (0.08 * progress); // Ken Burns
        ctx.translate(width / 2, height / 2);
        ctx.scale(scale, scale);
        ctx.translate(-width / 2, -height / 2);

        if (img) {
          const imgScale = Math.max(width / img.width, height / img.height);
          const x = (width - img.width * imgScale) / 2;
          const y = (height - img.height * imgScale) / 2;
          ctx.drawImage(img, x, y, img.width * imgScale, img.height * imgScale);
        } else {
          ctx.fillStyle = "#111"; 
          ctx.fillRect(0,0,width,height);
        }
        ctx.restore();

        if (effectiveDuration - elapsed < 500 && nextImg) {
          ctx.save();
          ctx.globalAlpha = 1 - ((effectiveDuration - elapsed) / 500);
          const imgScale = Math.max(width / nextImg.width, height / nextImg.height);
          const x = (width - nextImg.width * imgScale) / 2;
          const y = (height - nextImg.height * imgScale) / 2;
          ctx.drawImage(nextImg, x, y, nextImg.width * imgScale, nextImg.height * imgScale);
          ctx.restore();
        }

        await new Promise(r => setTimeout(r, 16));
      }
    }
    
    recorder.stop();
    await new Promise(r => { recorder.onstop = r; });
    
    const blob = new Blob(chunks, { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `storyboard_${Date.now()}.mp4`;
    a.click();
  };

  const handleDrop = (targetIndex: number) => {
    if (draggedIndex === null || draggedIndex === targetIndex) return;
    const newScenes = [...scenes];
    const [item] = newScenes.splice(draggedIndex, 1);
    newScenes.splice(targetIndex, 0, item);
    setScenes(newScenes);
    setDraggedIndex(null);
  };

  return (
    <div className="bg-gray-950 h-full font-sans text-gray-200 flex flex-col">
      <SettingsModal 
        isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} settings={settings} onSave={setSettings}
      />
      <Header 
        onReset={handleReset} settings={settings} setSettings={setSettings}
        isExporting={isExporting} onExport={handleExportVideo} onOpenSettings={() => setIsSettingsOpen(true)}
      />
      <main className="flex-1 flex flex-col md:flex-row overflow-hidden min-h-0">
        <div className="w-full md:w-[350px] bg-gray-900 border-r border-gray-800 flex flex-col z-10 shadow-xl shrink-0">
          <div className="p-4 flex-1 flex flex-col min-h-0 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700">
             <label className="text-gray-400 text-xs font-bold uppercase tracking-wider mb-2">脚本内容</label>
             <textarea
               className="bg-gray-800/50 text-gray-100 border border-gray-700 rounded-lg p-3 focus:ring-2 focus:ring-purple-500 focus:outline-none resize-none font-mono text-sm mb-6 h-32"
               placeholder="输入故事脚本..."
               value={script} onChange={(e) => setScript(e.target.value)}
             />
             <div className="flex-1 overflow-y-auto mb-4 min-h-[150px]">
                <div className="flex justify-between items-center mb-2">
                   <label className="text-purple-400 text-xs font-bold uppercase tracking-wider">🎭 角色 (自动提取)</label>
                   <button onClick={() => setCharacters([...characters, {id: Date.now().toString(), name: '新角色', description: ''}])} className="text-xs text-gray-400 hover:text-white">+ 添加</button>
                </div>
                {characters.length === 0 && <div className="text-gray-600 text-xs italic p-4 text-center border border-gray-800 border-dashed rounded">点击一键分析生成角色</div>}
                {characters.map(char => (
                   <CharacterEdit 
                     key={char.id} char={char} 
                     onUpdate={(c) => setCharacters(chars => chars.map(x => x.id === c.id ? c : x))}
                     onDelete={(id) => setCharacters(chars => chars.filter(x => x.id !== id))}
                   />
                ))}
             </div>
             <button onClick={handleAnalyze} disabled={isAnalyzing || !script.trim()} className={`w-full py-4 rounded-lg font-bold text-white transition-all flex items-center justify-center gap-2 shadow-lg shrink-0 mt-auto ${isAnalyzing ? 'bg-gray-700 cursor-not-allowed' : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:scale-[1.02]'}`}>
                {isAnalyzing ? <span className="animate-pulse">AI 分析生成中...</span> : <span>⚡ 一键分析脚本 & 生成</span>}
             </button>
          </div>
        </div>
        <div className="flex-1 bg-gray-950 flex flex-col min-h-0 relative">
           <div className="p-4 border-b border-gray-800 bg-gray-900/50 backdrop-blur flex justify-between items-center z-10">
             <span className="text-sm font-bold text-gray-300 flex items-center gap-2">预览 <span className="bg-gray-800 px-2 py-0.5 rounded-full text-xs text-gray-500">{scenes.length}</span></span>
           </div>
           <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-gray-700">
              {scenes.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-gray-700 select-none">
                   <div className="text-6xl mb-6 opacity-20 grayscale">🎬</div>
                   <p className="text-lg font-medium opacity-50">输入脚本开始创作</p>
                </div>
              ) : (
                scenes.map((scene, index) => (
                   <SceneCard 
                     key={scene.id} index={index} scene={scene}
                     onSelectImage={(sid, imgIdx) => setScenes(prev => prev.map(s => s.id === sid ? {...s, selectedImageIndex: imgIdx} : s))}
                     onRegenerateImages={handleRegenerateImages}
                     onDragStart={setDraggedIndex} onDragOver={(e) => e.preventDefault()} onDrop={() => handleDrop(index)}
                     onOpenLightbox={setLightboxImage}
                   />
                ))
              )}
           </div>
           {isExporting && (
             <div className="absolute inset-0 bg-black/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center p-8">
               <div className="w-full max-w-md bg-gray-900 rounded-2xl p-6 border border-gray-700 shadow-2xl">
                 <h3 className="text-white font-bold text-lg mb-2">正在导出视频</h3>
                 <p className="text-gray-400 text-sm font-mono text-center mb-4">{exportProgress}</p>
                 <div className="w-full bg-gray-800 rounded-full h-1 overflow-hidden"><div className="h-full bg-purple-500 animate-pulse w-full origin-left"></div></div>
               </div>
             </div>
           )}
        </div>
      </main>
      {lightboxImage && (
        <div className="fixed inset-0 z-[60] bg-black/95 flex items-center justify-center p-4 cursor-zoom-out backdrop-blur-md" onClick={() => setLightboxImage(null)}>
          <img src={lightboxImage} className="max-w-full max-h-full rounded shadow-2xl" />
        </div>
      )}
    </div>
  );
};

const App = () => (
  <ErrorBoundary>
    <AppContent />
  </ErrorBoundary>
);

const container = document.getElementById('app');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
