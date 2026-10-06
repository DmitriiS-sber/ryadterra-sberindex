/** DOM/data checks. This is not a substitute for browser layout or public-access QA. */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {parseHTML}=require(process.env.LINKEDOM_MODULE || 'linkedom');
const root=path.resolve(path.dirname(new URL(import.meta.url).pathname),'../dist');
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const {window}=parseHTML(html),{document}=window;
let width=1124;
Object.defineProperty(window.HTMLElement.prototype,'clientWidth',{get(){return width;}});
window.matchMedia=()=>({matches:true});
const context=vm.createContext({document,window,console,setTimeout,clearTimeout,navigator:{}});
const source=fs.readFileSync(path.join(root,'app.js'),'utf8');
vm.runInContext(source,context);
let checks=0;
function check(value,message){assert.ok(value,message);checks++;}
const payload=JSON.parse(document.getElementById('research-data').textContent);
const old=JSON.parse(fs.readFileSync(path.join(root,'assets/results.json'),'utf8')).validation;
for(const key of ['forecast','equal','origins','conditional_intervals','territory_comparison','detectors','null12','warnings','news','cases','case_paths']){
 check(JSON.stringify(payload[key])===JSON.stringify(old[key]),`Frozen source matches: ${key}`);
}
const ids=[...document.querySelectorAll('[id]')].map(n=>n.id);
check(ids.length===new Set(ids).size,'Unique document IDs');
for(const el of document.querySelectorAll('a[href],script[src],link[href]')){
 const href=el.getAttribute('href')||el.getAttribute('src');
 if(href.startsWith('#'))check(!!document.getElementById(href.slice(1)),`Anchor ${href}`);
 else if(!/^[a-z]+:/i.test(href))check(fs.existsSync(path.join(root,decodeURIComponent(href.split('#')[0]))),`Asset ${href}`);
}
for(const horizon of [1,3,6,12])for(const metric of ['mae_rub','rmse_rub','wape_pct']){
 document.querySelector(`[data-horizon="${horizon}"]`).click();
 const select=document.getElementById('metric-select');
 for(const opt of select.querySelectorAll('option'))opt.removeAttribute('selected');
 select.querySelector(`option[value="${metric}"]`).selected=true;
 select.dispatchEvent(new window.Event('change'));
 check(document.querySelectorAll('.model-row').length===9,`Nine models: ${horizon}/${metric}`);
 const best=payload.forecast.filter(r=>r.horizon===horizon).sort((a,b)=>a[metric]-b[metric])[0];
 const actual=vm.runInContext('state.metric',context);
 check(actual===metric,'Metric change handler');
 const expected=vm.runInContext(`modelNames[${JSON.stringify(best.model)}]`,context);
 check(document.getElementById('winning-model').textContent===expected,`Correct winner: ${horizon}/${metric}`);
 check(document.querySelectorAll('#forecast-table tbody tr').length===9,'Full metric table');
}
check(document.querySelectorAll('.detector-item').length===4,'Four detectors');
document.querySelector('[data-detection="real"]').click();
check(!document.getElementById('real-panel').hidden&&document.getElementById('synthetic-panel').hidden,'Real detector tab');
document.querySelector('[data-detection="synthetic"]').click();
check(document.getElementById('real-panel').hidden&&!document.getElementById('synthetic-panel').hidden,'Synthetic detector tab');
for(const viewport of [1440,834,390,320]){
 width=viewport-44;
 for(const territory of [322,319])for(const baseline of ['ridge','prophet','last_value'])for(const detector of ['ewma','cusum','page_hinkley','shewhart']){
  document.querySelector(`[data-territory="${territory}"]`).click();
  for(const [id,value] of [['baseline-select',baseline],['detector-select',detector]]){
   const select=document.getElementById(id);
   for(const opt of select.querySelectorAll('option'))opt.removeAttribute('selected');
   select.querySelector(`option[value="${value}"]`).selected=true;
   select.dispatchEvent(new window.Event('change'));
  }
  for(let month=0;month<12;month++){
   const range=document.getElementById('month-range');range.value=String(month);range.dispatchEvent(new window.Event('input'));
   check(!/NaN|Infinity|undefined/.test(document.getElementById('case-chart').innerHTML),'Finite chart coordinates');
   const row=payload.case_paths.find(r=>r.territory_id===territory&&r.date===`2024-${String(month+1).padStart(2,'0')}`);
   check((document.getElementById('case-actual').textContent==='Нет данных')===(row.actual===null),'Missing actual stays missing');
   check((document.getElementById('case-prediction').textContent==='Нет данных')===(row[baseline]===null),'Missing forecast stays missing');
  }
 }
}
document.querySelector('.menu-toggle').click();
check(document.querySelector('.menu-toggle').getAttribute('aria-expanded')==='true','Mobile menu opens');
const escape=new window.Event('keydown');Object.defineProperty(escape,'key',{value:'Escape'});document.dispatchEvent(escape);
check(document.querySelector('.menu-toggle').getAttribute('aria-expanded')==='false','Escape closes menu');
check(!document.documentElement.classList.contains('js-motion'),'Reduced motion has no reveal animation');
console.log(JSON.stringify({status:'passed',checks,coverage:'DOM logic, frozen data, local links, finite SVG at four input widths. Browser layout and anonymous production access are NOT verified.'},null,2));
