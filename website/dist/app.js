'use strict';
/** Offline-capable research presentation. All values come from the frozen result payload. */
const data=JSON.parse(document.getElementById('research-data').textContent);
const state={horizon:1,metric:'mae_rub',territory:322,baseline:'ridge',detector:'ewma',month:5};
const modelNames={last_value:'Последнее значение',seasonal_naive:'Сезонный наивный',damped_trend:'Затухающий тренд',prophet_auto:'Prophet · 5 точек',prophet_default:'Prophet · по умолчанию',prophet_monthly:'Prophet · сезонность',pooled_ridge:'Ridge',pooled_ridge_news:'Ridge + новости',chronos_bolt_tiny:'Chronos-Bolt Tiny'};
const methodNames={cusum:'CUSUM',page_hinkley:'Page–Hinkley',ewma:'EWMA',shewhart:'Shewhart'};
const baselineNames={ridge:'Ridge',prophet:'Prophet',last_value:'Последнее значение'};
const months=['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Ноябрь','Декабрь'];
const number=(value,digits=0)=>value===null||value===undefined||!Number.isFinite(Number(value))?'не определено':Number(value).toLocaleString('ru-RU',{minimumFractionDigits:digits,maximumFractionDigits:digits});
const byId=id=>document.getElementById(id);
function element(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function tableRow(values){const tr=element('tr');values.forEach((v,i)=>{const cell=element(i===0?'th':'td',v);if(i===0)cell.scope='row';tr.append(cell);});return tr;}
function pressed(attribute,value){document.querySelectorAll(`[${attribute}]`).forEach(button=>button.setAttribute('aria-pressed',String(button.getAttribute(attribute)===String(value))));}
function renderForecast(){
 const rows=data.forecast.filter(r=>r.horizon===state.horizon).sort((a,b)=>a[state.metric]-b[state.metric]);
 const best=rows[0],maximum=Math.max(...rows.map(r=>r[state.metric])),unit=state.metric==='wape_pct'?'%':'₽',digits=state.metric==='wape_pct'?2:0;
 pressed('data-horizon',state.horizon);
 const bars=byId('model-bars');bars.replaceChildren();
 rows.forEach((item,i)=>{
   const row=element('div',undefined,'model-row'+(i===0?' best':''));
   const label=element('span',modelNames[item.model],'model-name');
   const track=element('div',undefined,'bar-track');track.setAttribute('aria-hidden','true');
   const fill=element('div',undefined,'bar-fill');fill.style.width=`${Math.max(0,item[state.metric]/maximum*100)}%`;track.append(fill);
   row.append(label,track,element('span',number(item[state.metric],digits),'bar-value'));bars.append(row);
 });
 byId('winning-model').textContent=modelNames[best.model];byId('winning-value').textContent=number(best[state.metric],digits)+' '+unit;
 byId('sample-size').textContent=`Дат прогноза: ${best.origins}. Тестовых целей: ${number(best.n)}.`;
 byId('horizon-note').textContent=state.horizon===12?'На годовом горизонте доступна одна дата отсечения. Устойчивость преимущества во времени пока не проверена.':'Результаты Prophet с пятью точками изменения и Prophet по умолчанию почти совпадают. Превосходство зависит от даты и территории.';
 byId('metric-definition').textContent={mae_rub:'MAE — средняя абсолютная ошибка. Доступные тестовые пары имеют равный вес.',rmse_rub:'RMSE — корень из средней квадратичной ошибки. Сильнее реагирует на большие промахи.',wape_pct:'WAPE — сумма абсолютных ошибок, делённая на сумму фактических расходов, в процентах.'}[state.metric];
 byId('chart-caption').textContent=`Горизонт ${state.horizon} мес. · ${state.metric==='mae_rub'?'MAE':state.metric==='rmse_rub'?'RMSE':'WAPE'}, ${unit}. Общая доступная выборка для всех девяти конфигураций. Значения округлены только для отображения.`;
 const tbody=document.querySelector('#forecast-table tbody');tbody.replaceChildren();
 rows.forEach(r=>{const eq=data.equal.find(v=>v.model===r.model&&v.horizon===r.horizon);const row=tableRow([modelNames[r.model],number(r.mae_rub,2),number(eq.equal_territory_mae_rub,2),number(r.rmse_rub,2),number(r.wape_pct,2),number(r.r2,3)]);if(r.model===best.model)row.className='winner';tbody.append(row);});
}
function renderDetectors(){
 const target=byId('detector-metrics');
 Object.keys(methodNames).forEach(method=>{
   const r=data.detectors.find(v=>v.model===method&&v.kind==='step'),n=data.detectors.find(v=>v.model===method&&v.kind==='null');
   const article=element('article',undefined,'detector-item');article.append(element('h3',methodNames[method]),element('div',number(r.recall_12m_pct,1)+'%','recall'),element('div','полнота за 12 месяцев','metric-label'));
   const details=element('div',undefined,'detector-details');
   const delay=element('p');delay.append(element('strong',number(r.median_delay_months)+' мес.'),document.createTextNode(' — медианная задержка'));
   const error=element('p');error.append(element('strong',number(n.any_alarm_pct,1)+'%'),document.createTextNode(' — null-рядов с тревогой'));
   details.append(delay,error);article.append(details);target.append(article);
 });
 const body=document.querySelector('#real-table tbody');data.null12.forEach(r=>body.append(tableRow([methodNames[r.model],number(data.news.find(n=>n.method===r.model&&!n.use_news).territories_with_candidate_alert),number(r.any_alarm_pct,1),number(r.alarm_wilson_lower_pct,1)+'–'+number(r.alarm_wilson_upper_pct,1)+'%'])));
}
function svgElement(tag,attributes,text){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attributes||{}).forEach(([key,value])=>node.setAttribute(key,String(value)));if(text!==undefined)node.textContent=text;return node;}
function pathFor(rows,key,x,y){let pen=false;return rows.map((r,i)=>{const value=r[key];if(value===null||value===undefined||!Number.isFinite(Number(value))){pen=false;return '';}const move=pen?'L':'M';pen=true;return `${move}${x(i).toFixed(2)},${y(value).toFixed(2)}`;}).join(' ');}
function renderCase(){
 const rows=data.case_paths.filter(r=>r.territory_id===state.territory);const record=data.cases.find(r=>r.territory_id===state.territory);const sample=rows[state.month];
 pressed('data-territory',state.territory);byId('baseline-legend').textContent=baselineNames[state.baseline];
 const chart=byId('case-chart');const width=Math.max(280,Math.round(chart.clientWidth)),height=width<550?295:360;
 const left=width<550?48:62,right=15,top=30,bottom=38;
 const values=rows.flatMap(r=>[r.actual,r[state.baseline]]).filter(v=>v!==null&&Number.isFinite(v));
 const min=Math.floor(Math.min(...values)/10000)*10000,max=Math.ceil(Math.max(...values)/10000)*10000;
 const x=i=>left+i*(width-left-right)/11,y=value=>top+(max-value)/(max-min)*(height-top-bottom);
 const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-labelledby':'case-svg-title case-svg-description'});
 svg.append(svgElement('title',{id:'case-svg-title'},`Территория ${state.territory}: фактические расходы и ${baselineNames[state.baseline]}, 2024 год`),svgElement('desc',{id:'case-svg-description'},'Чёрная линия — фактические расходы. Синяя пунктирная — выбранный однопериодный прогноз. Золотая вертикальная линия — первый сигнал выбранного детектора. Значения доступных месяцев можно прочитать с помощью ползунка под графиком.'));
 for(let value=min;value<=max;value+=10000){svg.append(svgElement('line',{x1:left,y1:y(value),x2:width-right,y2:y(value),stroke:'#dcdce3','stroke-width':1}),svgElement('text',{x:left-10,y:y(value)+4,'text-anchor':'end'},number(value/1000)+' тыс.'));}
 for(let i=0;i<12;i++){if(width<550&&![0,3,6,9,11].includes(i))continue;svg.append(svgElement('text',{x:x(i),y:height-10,'text-anchor':'middle'},months[i].slice(0,3).toLowerCase()));}
 const alarm=record.corrected_alerts.find(a=>a.model===state.detector);
 if(alarm){const i=Number(alarm.first_alarm.slice(5))-1;svg.append(svgElement('line',{x1:x(i),x2:x(i),y1:top-8,y2:height-bottom,stroke:'#a76514','stroke-width':1.5,'stroke-dasharray':'4 5'}),svgElement('text',{x:Math.min(x(i)+7,width-68),y:16,fill:'#8c5514'},'Сигнал'));}
 svg.append(svgElement('path',{d:pathFor(rows,state.baseline,x,y),fill:'none',stroke:'#0067d8','stroke-width':2.6,'stroke-dasharray':'7 5','stroke-linejoin':'round','stroke-linecap':'round'}),svgElement('path',{d:pathFor(rows,'actual',x,y),fill:'none',stroke:'#17171b','stroke-width':3,'stroke-linejoin':'round','stroke-linecap':'round'}));
 rows.forEach((r,i)=>{if(r.actual!==null)svg.append(svgElement('circle',{cx:x(i),cy:y(r.actual),r:3.5,fill:'#17171b'}));});
 svg.append(svgElement('line',{x1:x(state.month),x2:x(state.month),y1:top,y2:height-bottom,stroke:'#7c7c87','stroke-width':1,opacity:.5}));
 if(sample.actual!==null)svg.append(svgElement('circle',{cx:x(state.month),cy:y(sample.actual),r:6,fill:'#fff',stroke:'#17171b','stroke-width':3}));
 if(sample[state.baseline]!==null)svg.append(svgElement('circle',{cx:x(state.month),cy:y(sample[state.baseline]),r:5,fill:'#fff',stroke:'#0067d8','stroke-width':2.5}));
 chart.replaceChildren(svg);
 const month=months[state.month]+' 2024';byId('month-label').textContent=month;byId('month-range').setAttribute('aria-valuetext',month);
 byId('case-actual').textContent=sample.actual===null?'Нет данных':number(sample.actual)+' ₽';
 byId('case-prediction').textContent=sample[state.baseline]===null?'Нет данных':number(sample[state.baseline])+' ₽';
 const diff=sample.actual!==null&&sample[state.baseline]!==null?100*(sample.actual/sample[state.baseline]-1):null;
 byId('case-deviation').textContent=diff===null?'Нет данных':(diff>0?'+':'')+number(diff,1)+'%';
 byId('signal-description').textContent=(alarm?`${methodNames[state.detector]}: первый сигнал — ${months[Number(alarm.first_alarm.slice(5))-1].toLowerCase()} 2024. `:`${methodNames[state.detector]}: в исправленном протоколе сигнал для этой территории не получен. `)+'Все сигналы здесь рассчитаны по ошибкам Ridge. Переключение базовой линии помогает оценить зависимость отклонения от модели.';
}
function setupNavigation(){
 const button=document.querySelector('.menu-toggle'),nav=byId('site-nav');
 function close(){nav.classList.remove('open');button.setAttribute('aria-expanded','false');}
 button.addEventListener('click',()=>{const open=button.getAttribute('aria-expanded')!=='true';button.setAttribute('aria-expanded',String(open));nav.classList.toggle('open',open);});
 nav.querySelectorAll('a').forEach(a=>a.addEventListener('click',close));
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&button.getAttribute('aria-expanded')==='true'){close();button.focus();}});
 document.addEventListener('click',e=>{if(!document.querySelector('.site-header').contains(e.target))close();});
}
async function copyCommands(){
 const text=byId('run-command').textContent;const status=byId('copy-status');
 try{if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(text);else{const input=element('textarea');input.value=text;input.style.position='fixed';input.style.left='-9999px';document.body.append(input);input.select();const ok=document.execCommand('copy');input.remove();byId('copy-command').focus();if(!ok)throw new Error('Clipboard unavailable');}status.textContent='Команды скопированы.';}catch{status.textContent='Автоматическое копирование недоступно. Выделите команды в блоке и скопируйте их.';}
}
function init(){
 renderForecast();renderDetectors();renderCase();setupNavigation();
 document.querySelectorAll('[data-horizon]').forEach(b=>b.addEventListener('click',()=>{state.horizon=Number(b.dataset.horizon);renderForecast();}));
 byId('metric-select').addEventListener('change',e=>{state.metric=e.target.value;renderForecast();});
 document.querySelectorAll('[data-detection]').forEach(b=>b.addEventListener('click',()=>{const real=b.dataset.detection==='real';pressed('data-detection',b.dataset.detection);byId('real-panel').hidden=!real;byId('synthetic-panel').hidden=real;}));
 document.querySelectorAll('[data-territory]').forEach(b=>b.addEventListener('click',()=>{state.territory=Number(b.dataset.territory);state.month=state.territory===322?5:6;byId('month-range').value=String(state.month);renderCase();}));
 byId('baseline-select').addEventListener('change',e=>{state.baseline=e.target.value;renderCase();});byId('detector-select').addEventListener('change',e=>{state.detector=e.target.value;renderCase();});
 byId('month-range').addEventListener('input',e=>{state.month=Number(e.target.value);renderCase();});
 byId('copy-command').addEventListener('click',copyCommands);
 let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(renderCase,100);});
 if(!window.matchMedia('(prefers-reduced-motion: reduce)').matches&&'IntersectionObserver'in window){document.documentElement.classList.add('js-motion');const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){entry.target.classList.add('visible');observer.unobserve(entry.target);}}),{threshold:.06});document.querySelectorAll('.reveal').forEach(node=>observer.observe(node));}
}
init();
