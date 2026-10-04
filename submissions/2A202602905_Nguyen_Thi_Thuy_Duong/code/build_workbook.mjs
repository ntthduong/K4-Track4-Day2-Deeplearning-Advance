import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile,FileBlob} from '@oai/artifact-tool';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const base=root+'/evidence/workbook-build';
await fs.mkdir(base,{recursive:true});
const d=JSON.parse(await fs.readFile(root+'/evidence/workbook-data.json','utf8'));
d.output=root+'/results.xlsx';d.submission_output=d.output;
if(d.original){
 const old=await SpreadsheetFile.importXlsx(await FileBlob.load(d.original));
 console.log((await old.inspect({kind:'workbook,sheet,table',maxChars:2000,tableMaxRows:2,tableMaxCols:5})).ndjson);
 const image=await old.render({sheetName:'Final',range:'A1:H8',scale:1,format:'png'});
 await fs.writeFile(base+'/original_final.png',new Uint8Array(await image.arrayBuffer()));
}
if(process.argv.includes('--inspect'))process.exit(0);
const wb=Workbook.create();
const sheets={};
for(const name of ['Backbones','Training','Inference','Final','PerClass','Latency','Summary'])sheets[name]=wb.worksheets.add(name);
const col=i=>{let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s};
for(const [name,info] of Object.entries(d.sheets)){
 const sh=sheets[name]; const nc=info.headers.length, end=col(nc-1), last=4+info.rows.length;
 sh.showGridLines=false;sh.tabColor=name==='Summary'?'#0F766E':'#27475D';
 sh.getRange(`A1:${end}${last+3}`).format.font={name:'Arial',size:10,color:'#172D3A'};
 sh.getRange(`A1:${end}1`).merge();sh.getRange('A1').values=[[info.title]];
 sh.getRange('A1').format.font={name:'Arial',size:14,bold:true,color:'#172D3A'};
 sh.getRange(`A2:${end}2`).merge();sh.getRange('A2').values=[[info.note]];
 sh.getRange(`A2:${end}2`).format={rowHeight:32,wrapText:true,borders:{bottom:{style:'thin',color:'#0F766E'}},font:{name:'Arial',size:9,color:'#49626D'}};
 sh.getRange(`A4:${end}4`).values=[info.headers];
 sh.getRange(`A4:${end}4`).format={fill:'#27475D',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42,verticalAlignment:'center',horizontalAlignment:'center'};
 if(info.rows.length)sh.getRange(`A5:${end}${last}`).values=info.rows;
 sh.getRange(`A5:${end}${last}`).format.rowHeight=info.rowHeight||26;
 sh.getRange(`A5:${end}${last}`).format.verticalAlignment='center';
 for(let i=0;i<nc;i++){
  sh.getRange(`${col(i)}4:${col(i)}${last}`).format.columnWidth=info.widths?.[i]||16;
  if(info.numeric?.includes(i))sh.getRange(`${col(i)}5:${col(i)}${last}`).setNumberFormat('0.0000');
  if(info.integer?.includes(i))sh.getRange(`${col(i)}5:${col(i)}${last}`).setNumberFormat('#,##0');
  if(info.wrap?.includes(i))sh.getRange(`${col(i)}5:${col(i)}${last}`).format.wrapText=true;
 }
 for(let r=5;r<=last;r++)if(r%2===0)sh.getRange(`A${r}:${end}${r}`).format.fill='#F2F6F7';
 for(const [cell,f] of Object.entries(info.formulas||{}))sh.getRange(cell).formulas=[[f]];
 for(const r of info.highlight||[])sh.getRange(`A${r}:${end}${r}`).format={fill:'#DEF0E8',font:{name:'Arial',size:10,bold:true,color:'#164E38'}};
 sh.freezePanes.freezeRows(4);sh.freezePanes.freezeColumns(1);
 if(info.rows.length)sh.tables.add(`A4:${end}${last}`,true,`${name}Results`);
 if(info.footer){sh.getRange(`A${last+2}:${end}${last+2}`).merge();sh.getRange(`A${last+2}`).values=[[info.footer]];sh.getRange(`A${last+2}:${end}${last+2}`).format={wrapText:true,rowHeight:40,font:{name:'Arial',size:9,color:'#49626D'}};}
}
wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},maxChars:3000});
console.log(errors.ndjson);
console.log((await wb.inspect({kind:'region',sheetId:'Final',range:'A5:K13',maxChars:5000,tableMaxRows:10,tableMaxCols:11})).ndjson);
// Confirm formulas respond to a changed recorded input, then restore the source.
const original=sheets.Training.getRange('F5').values;
const prior=sheets.Training.getRange('H11').values[0][0];
sheets.Training.getRange('F5').values=[[original[0][0]+0.01]];wb.recalculate();
const changed=sheets.Training.getRange('H11').values[0][0];
if(Math.abs((prior-changed)-0.01)>1e-9)throw Error('Training delta recalculation failed');
sheets.Training.getRange('F5').values=original;wb.recalculate();
for(const name of Object.keys(sheets)){
 const img=await wb.render({sheetName:name,range:d.sheets[name].preview||`A1:${col(d.sheets[name].headers.length-1)}${Math.min(d.sheets[name].rows.length+7,24)}`,scale:1,format:'png'});
 await fs.writeFile(base+`/preview_${name}.png`,new Uint8Array(await img.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(d.output);

await fs.writeFile(base+'/verification.json',JSON.stringify({formulaErrors:errors.ndjson,recalculationPassed:true,output:d.output},null,2));
console.log('WORKBOOK_READY',d.output);
