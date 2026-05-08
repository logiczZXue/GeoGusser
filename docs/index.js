// 全局变量
let map = null;
let panorama = null;
let markers = [];
const DEFAULT_API_KEY = '';
const DEFAULT_PANO_ID = 'sVY13Eno_UlGbCn1wIPFcw'; // 默认的pano_id

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM加载完成，初始化地图组件...');
  
  // 获取DOM元素
  const apiKeyInput = document.getElementById('api-key-input');
  const loadMapBtn = document.getElementById('load-map-btn');
  const testApiBtn = document.getElementById('test-api-btn');
  const loadDefaultMapBtn = document.getElementById('load-default-map-btn');
  const panoIdInput = document.getElementById('pano-id-input');
  const loadPanoBtn = document.getElementById('load-pano-btn');
  const mapContainer = document.getElementById('map-container');
  const mapError = document.getElementById('map-error');
  const errorCloseBtn = mapError.querySelector('.delete');
  
  // 检查元素是否存在
  if (!apiKeyInput || !loadMapBtn || !mapContainer || !mapError) {
    console.error('无法找到必要的DOM元素');
    return;
  }
  
  // 设置默认API密钥（可选）
  apiKeyInput.value = DEFAULT_API_KEY;
  
  // 加载地图按钮点击事件
  loadMapBtn.addEventListener('click', function() {
    console.log('加载地图按钮被点击');
    const apiKey = apiKeyInput.value.trim();
    
    if (!apiKey) {
      showError('请输入有效的API密钥');
      return;
    }
    
    console.log('使用API密钥:', apiKey);
    
    // 隐藏错误信息
    mapError.style.display = 'none';
    
    // 动态加载Google Maps API
    loadGoogleMapsAPI(apiKey);
  });
  
  // 测试API密钥按钮点击事件
  if (testApiBtn) {
    testApiBtn.addEventListener('click', function() {
      console.log('测试API密钥按钮被点击');
      const apiKey = apiKeyInput.value.trim();
      
      if (!apiKey) {
        showError('请输入有效的API密钥');
        return;
      }
      
      console.log('测试API密钥:', apiKey);
      
      // 隐藏错误信息
      mapError.style.display = 'none';
      
      // 测试API密钥
      testApiKey(apiKey);
    });
  }
  
  // 加载默认地图按钮点击事件
  if (loadDefaultMapBtn) {
    loadDefaultMapBtn.addEventListener('click', function() {
      console.log('加载默认地图按钮被点击');
      
      // 隐藏错误信息
      mapError.style.display = 'none';
      
      // 使用默认API密钥加载地图
      loadGoogleMapsAPI(DEFAULT_API_KEY);
    });
  }
  
  // 加载特定全景图按钮点击事件
  if (loadPanoBtn) {
    loadPanoBtn.addEventListener('click', function() {
      console.log('加载特定全景图按钮被点击');
      const apiKey = apiKeyInput.value.trim();
      const panoId = panoIdInput.value.trim();
      
      if (!apiKey) {
        showError('请输入有效的API密钥');
        return;
      }
      
      if (!panoId) {
        showError('请输入有效的全景图ID');
        return;
      }
      
      console.log('使用API密钥:', apiKey, '加载全景图:', panoId);
      
      // 隐藏错误信息
      mapError.style.display = 'none';
      
      // 设置当前要加载的pano_id
      window.CURRENT_PANO_ID = panoId;
      
      // 动态加载Google Maps API
      loadGoogleMapsAPI(apiKey);
    });
  }
  
  // 关闭错误信息按钮
  errorCloseBtn.addEventListener('click', function() {
    mapError.style.display = 'none';
  });
  
  // 按Enter键也可以加载地图
  apiKeyInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      console.log('按下Enter键，触发加载地图');
      loadMapBtn.click();
    }
  });
  
  // 自动加载地图（可选）
  // loadMapBtn.click();
});

// 测试API密钥
function testApiKey(apiKey) {
  console.log('开始测试API密钥...');
  
  // 创建一个图片元素来测试API密钥
  const img = new Image();
  img.onload = function() {
    console.log('API密钥有效');
    showSuccess('API密钥有效！点击"Load Map"按钮加载地图。');
  };
  
  img.onerror = function() {
    console.error('API密钥无效');
    showError('API密钥无效。请检查您的API密钥并重试。');
  };
  
  // 使用Google Maps Static API测试密钥
  img.src = `https://maps.googleapis.com/maps/api/staticmap?center=0,0&zoom=1&size=1x1&key=${apiKey}`;
}

// 显示成功信息
function showSuccess(message) {
  console.log('成功:', message);
  const mapError = document.getElementById('map-error');
  const errorMessage = mapError.querySelector('p');
  
  // 更改样式为成功
  mapError.className = 'notification is-success';
  
  errorMessage.textContent = message;
  mapError.style.display = 'block';
}

// 动态加载Google Maps API
function loadGoogleMapsAPI(apiKey) {
  console.log('开始加载Google Maps API...');
  
  // 检查是否已经加载了API
  if (window.google && window.google.maps) {
    console.log('Google Maps API已经加载，直接初始化地图');
    initPano();
    return;
  }
  
  // 创建script元素
  const script = document.createElement('script');
  script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&callback=initPano&v=weekly`;
  script.defer = true;
  script.async = true;
  
  // 处理加载错误
  script.onerror = function() {
    console.error('加载Google Maps API失败');
    showError('加载Google Maps API失败。请检查您的API密钥和网络连接。');
  };
  
  // 添加到文档
  document.head.appendChild(script);
  console.log('Google Maps API脚本已添加到文档');
}

// 显示错误信息
function showError(message) {
  console.error('错误:', message);
  const mapError = document.getElementById('map-error');
  const errorMessage = mapError.querySelector('p');
  
  // 确保样式为错误
  mapError.className = 'notification is-danger';
  
  errorMessage.textContent = message;
  mapError.style.display = 'block';
}

// 初始化全景图
function initPano() {
  console.log('初始化全景图...');
  try {
    // 显示地图容器
    const mapContainer = document.getElementById('map-container');
    if (!mapContainer) {
      console.error('找不到地图容器元素');
      return;
    }
    
    mapContainer.style.display = 'block';
    
    // 清除之前的标记
    if (markers.length > 0) {
      markers.forEach(marker => marker.setMap(null));
      markers = [];
    }
    
    // 创建地图
    const mapElement = document.getElementById("map");
    if (!mapElement) {
      console.error('找不到地图元素');
      return;
    }
    
    console.log('创建地图...');
    map = new google.maps.Map(mapElement, {
      center: { lat: 0, lng: 0 },
      zoom: 2,
      streetViewControl: false,
      mapTypeControl: false,
      fullscreenControl: false,
    });

    // 创建全景图
    console.log('创建全景图...');
    panorama = new google.maps.StreetViewPanorama(
      mapElement,
      {
        position: { lat: 0, lng: 0 },
        pov: { heading: 165, pitch: 0 },
        zoom: 1,
        addressControl: false,
        showRoadLabels: false,
        zoomControl: true,
        moveControl: true,
        panControl: true,
        enableCloseButton: false,
        pano: window.CURRENT_PANO_ID || DEFAULT_PANO_ID // 使用输入的pano_id或默认值
      }
    );

    map.setStreetView(panorama);
    console.log('地图和全景图设置完成，使用pano_id:', window.CURRENT_PANO_ID || DEFAULT_PANO_ID);
    
    // 添加示例位置标记
    addLocationMarkers();
    
  } catch (error) {
    console.error('初始化地图时出错:', error);
    showError('初始化地图时出错。请检查您的API密钥并重试。');
  }
}

// 添加位置标记
function addLocationMarkers() {
  console.log('添加位置标记...');
  // 示例位置
  const locations = [
    { lat: 48.8584, lng: 2.2945, name: "Eiffel Tower" },
    { lat: 40.7484, lng: -73.9857, name: "Empire State Building" },
    { lat: 35.6762, lng: 139.6503, name: "Tokyo Tower" },
  ];

  // 为每个位置添加标记
  locations.forEach((location) => {
    const marker = new google.maps.Marker({
      position: { lat: location.lat, lng: location.lng },
      map: map,
      title: location.name,
    });
    
    // 添加点击事件，点击标记时切换到该位置的全景图
    marker.addListener('click', function() {
      console.log('点击标记:', location.name);
      panorama.setPosition({ lat: location.lat, lng: location.lng });
      map.setStreetView(panorama);
    });
    
    markers.push(marker);
  });
  
  console.log('位置标记添加完成');
} 