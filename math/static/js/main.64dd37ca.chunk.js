(this["webpackJsonpkha-app"]=this["webpackJsonpkha-app"]||[]).push([[0],{19:function(e,t,n){},20:function(e,t){var n;!function(e){"use strict";var t=function(){function e(){this._dropEffect="move",this._effectAllowed="all",this._data={}}return Object.defineProperty(e.prototype,"dropEffect",{get:function(){return this._dropEffect},set:function(e){this._dropEffect=e},enumerable:!0,configurable:!0}),Object.defineProperty(e.prototype,"effectAllowed",{get:function(){return this._effectAllowed},set:function(e){this._effectAllowed=e},enumerable:!0,configurable:!0}),Object.defineProperty(e.prototype,"types",{get:function(){return Object.keys(this._data)},enumerable:!0,configurable:!0}),e.prototype.clearData=function(e){null!=e?delete this._data[e]:this._data=null},e.prototype.getData=function(e){return this._data[e]||""},e.prototype.setData=function(e,t){this._data[e]=t},e.prototype.setDragImage=function(e,t,r){var a=n._instance;a._imgCustom=e,a._imgOffset={x:t,y:r}},e}();e.DataTransfer=t;var n=function(){function e(){if(this._lastClick=0,e._instance)throw"DragDropTouch instance already created.";var t=!1;if(document.addEventListener("test",(function(){}),{get passive(){return t=!0,!0}}),"ontouchstart"in document){var n=document,r=this._touchstart.bind(this),a=this._touchmove.bind(this),o=this._touchend.bind(this),i=!!t&&{passive:!1,capture:!1};n.addEventListener("touchstart",r,i),n.addEventListener("touchmove",a,i),n.addEventListener("touchend",o),n.addEventListener("touchcancel",o)}}return e.getInstance=function(){return e._instance},e.prototype._touchstart=function(t){var n=this;if(this._shouldHandle(t)){if(Date.now()-this._lastClick<e._DBLCLICK&&this._dispatchEvent(t,"dblclick",t.target))return t.preventDefault(),void this._reset();this._reset();var r=this._closestDraggable(t.target);r&&(this._dispatchEvent(t,"mousemove",t.target)||this._dispatchEvent(t,"mousedown",t.target)||(this._dragSource=r,this._ptDown=this._getPoint(t),this._lastTouch=t,t.preventDefault(),setTimeout((function(){n._dragSource==r&&null==n._img&&n._dispatchEvent(t,"contextmenu",r)&&n._reset()}),e._CTXMENU),e._ISPRESSHOLDMODE&&(this._pressHoldInterval=setTimeout((function(){n._isDragEnabled=!0,n._touchmove(t)}),e._PRESSHOLDAWAIT))))}},e.prototype._touchmove=function(e){if(this._shouldCancelPressHoldMove(e))this._reset();else if(this._shouldHandleMove(e)||this._shouldHandlePressHoldMove(e)){var t=this._getTarget(e);if(this._dispatchEvent(e,"mousemove",t))return this._lastTouch=e,void e.preventDefault();this._dragSource&&!this._img&&this._shouldStartDragging(e)&&(this._dispatchEvent(e,"dragstart",this._dragSource),this._createImage(e),this._dispatchEvent(e,"dragenter",t)),this._img&&(this._lastTouch=e,e.preventDefault(),t!=this._lastTarget&&(this._dispatchEvent(this._lastTouch,"dragleave",this._lastTarget),this._dispatchEvent(e,"dragenter",t),this._lastTarget=t),this._moveImage(e),this._isDropZone=this._dispatchEvent(e,"dragover",t))}},e.prototype._touchend=function(e){if(this._shouldHandle(e)){if(this._dispatchEvent(this._lastTouch,"mouseup",e.target))return void e.preventDefault();this._img||(this._dragSource=null,this._dispatchEvent(this._lastTouch,"click",e.target),this._lastClick=Date.now()),this._destroyImage(),this._dragSource&&(e.type.indexOf("cancel")<0&&this._isDropZone&&this._dispatchEvent(this._lastTouch,"drop",this._lastTarget),this._dispatchEvent(this._lastTouch,"dragend",this._dragSource),this._reset())}},e.prototype._shouldHandle=function(e){return e&&!e.defaultPrevented&&e.touches&&e.touches.length<2},e.prototype._shouldHandleMove=function(t){return!e._ISPRESSHOLDMODE&&this._shouldHandle(t)},e.prototype._shouldHandlePressHoldMove=function(t){return e._ISPRESSHOLDMODE&&this._isDragEnabled&&t&&t.touches&&t.touches.length},e.prototype._shouldCancelPressHoldMove=function(t){return e._ISPRESSHOLDMODE&&!this._isDragEnabled&&this._getDelta(t)>e._PRESSHOLDMARGIN},e.prototype._shouldStartDragging=function(t){var n=this._getDelta(t);return n>e._THRESHOLD||e._ISPRESSHOLDMODE&&n>=e._PRESSHOLDTHRESHOLD},e.prototype._reset=function(){this._destroyImage(),this._dragSource=null,this._lastTouch=null,this._lastTarget=null,this._ptDown=null,this._isDragEnabled=!1,this._isDropZone=!1,this._dataTransfer=new t,clearInterval(this._pressHoldInterval)},e.prototype._getPoint=function(e,t){return e&&e.touches&&(e=e.touches[0]),{x:t?e.pageX:e.clientX,y:t?e.pageY:e.clientY}},e.prototype._getDelta=function(t){if(e._ISPRESSHOLDMODE&&!this._ptDown)return 0;var n=this._getPoint(t);return Math.abs(n.x-this._ptDown.x)+Math.abs(n.y-this._ptDown.y)},e.prototype._getTarget=function(e){for(var t=this._getPoint(e),n=document.elementFromPoint(t.x,t.y);n&&"none"==getComputedStyle(n).pointerEvents;)n=n.parentElement;return n},e.prototype._createImage=function(t){this._img&&this._destroyImage();var n=this._imgCustom||this._dragSource;if(this._img=n.cloneNode(!0),this._copyStyle(n,this._img),this._img.style.top=this._img.style.left="-9999px",!this._imgCustom){var r=n.getBoundingClientRect(),a=this._getPoint(t);this._imgOffset={x:a.x-r.left,y:a.y-r.top},this._img.style.opacity=e._OPACITY.toString()}this._moveImage(t),document.body.appendChild(this._img)},e.prototype._destroyImage=function(){this._img&&this._img.parentElement&&this._img.parentElement.removeChild(this._img),this._img=null,this._imgCustom=null},e.prototype._moveImage=function(e){var t=this;requestAnimationFrame((function(){if(t._img){var n=t._getPoint(e,!0),r=t._img.style;r.position="absolute",r.pointerEvents="none",r.zIndex="999999",r.left=Math.round(n.x-t._imgOffset.x)+"px",r.top=Math.round(n.y-t._imgOffset.y)+"px"}}))},e.prototype._copyProps=function(e,t,n){for(var r=0;r<n.length;r++){var a=n[r];e[a]=t[a]}},e.prototype._copyStyle=function(t,n){if(e._rmvAtts.forEach((function(e){n.removeAttribute(e)})),t instanceof HTMLCanvasElement){var r=t,a=n;a.width=r.width,a.height=r.height,a.getContext("2d").drawImage(r,0,0)}for(var o=getComputedStyle(t),i=0;i<o.length;i++){var c=o[i];c.indexOf("transition")<0&&(n.style[c]=o[c])}n.style.pointerEvents="none";for(i=0;i<t.children.length;i++)this._copyStyle(t.children[i],n.children[i])},e.prototype._dispatchEvent=function(t,n,r){if(t&&r){var a=document.createEvent("Event"),o=t.touches?t.touches[0]:t;return a.initEvent(n,!0,!0),a.button=0,a.which=a.buttons=1,this._copyProps(a,t,e._kbdProps),this._copyProps(a,o,e._ptProps),a.dataTransfer=this._dataTransfer,r.dispatchEvent(a),a.defaultPrevented}return!1},e.prototype._closestDraggable=function(e){for(;e;e=e.parentElement)if(e.hasAttribute("draggable")&&e.draggable)return e;return null},e}();n._instance=new n,n._THRESHOLD=5,n._OPACITY=.5,n._DBLCLICK=500,n._CTXMENU=900,n._ISPRESSHOLDMODE=!1,n._PRESSHOLDAWAIT=400,n._PRESSHOLDMARGIN=25,n._PRESSHOLDTHRESHOLD=0,n._rmvAtts="id,class,style,draggable".split(","),n._kbdProps="altKey,ctrlKey,metaKey,shiftKey".split(","),n._ptProps="pageX,pageY,clientX,clientY,screenX,screenY".split(","),e.DragDropTouch=n}(n||(n={}))},23:function(e,t,n){},24:function(e,t,n){},25:function(e,t,n){},26:function(e,t,n){"use strict";n.r(t);var r,a,o=n(1),i=n.n(o),c=n(13),s=n.n(c),u=(n(19),n(20),n(7)),l=n(11),d=n(14),h=n.n(d),p=n(6);(a=r||(r={})).IMAGES=["character-0.png","character-1.svg","character-2.svg","character-3.svg","character-4.svg","character-5.png","character-6.png","character-7.png","character-8.png","character-9.png","character-10.png","character-11.png","character-12.png","character-13.png","character-14.png","character-15.png","character-16.png","character-17.png","character-18.png","character-19.png","character-20.png","character-21.png","character-22.png","character-23.png","character-24.png","character-25.png","character-26.png","character-27.svg","character-28.png","character-29.png","character-30.png","character-31.png","character-32.png","character-33.png","character-34.png","character-35.png","character-36.png","edmond-elephant.png","suzy-sheep.png","chloe-pig.png","alexander-pig.png"],a.WILD_ANIMALS=["alligator","antelope","bear","camel","chimpanzee","coyote","crab","crocodile","dolphin","elephant","giraffe","gorilla","jellyfish","koala","leopard","lion","monkey","owl","panda","raccoon","shark","starfishes","walrus","wolf","woodpecker","zebra","badger","bat","deer","elk","fox","frog","hare","hedgehog","kangaroo","lizard","mole","otter","rabbit","rat","reindeer","snake","squirrel","toad"],a.SIZE=20,a.MIN=10,a.MAX=100;var g,f=n(3),v=n.n(f),m=n(2);!function(e){var t=v.a.mark(n);e.trace=function(e){var t="".concat((window.performance.now()/1e3).toFixed(3),": ").concat(String(e),"\n");console.warn(t);var n=document.getElementById("log");n&&(n.innerText=t+n.innerText)},e.requestFullscreen=function(e){var t=e.requestFullscreen||e.webkitRequestFullscreen||e.mozRequestFullScreen||e.msRequestFullscreen;t&&t.apply(e)};function n(e){var n,r;return v.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:n=Object(m.a)(Array(e)).map((function(e,t){return t})),r=[];case 2:return 0===r.length&&(r=Object(m.a)(n).sort((function(){return Math.random()-.5}))),t.next=6,r.pop();case 6:t.next=2;break;case 8:case"end":return t.stop()}}),t)}e.shuffleGenerator=n;e.random=function(e,t){return~~(Math.random()*(t-e))+e},e.cloneDigit=function(e,t,n){var r=e.cloneNode(!0),a=r.innerText===t;return r.style.background=a?"green":"red",r.ondragend=function(e){var t=document.createElement("div");t.className="digit-panel",e.target.replaceWith(t)},n(~~a-1),r},e.cloneLetter=function(e,t,n){var r=e.cloneNode(!0),a=r.innerText===t;return r.style.background=a?"green":"red",r.ondragend=function(e){var t=document.createElement("div");t.className="letter-panel",e.target.replaceWith(t)},n(~~a-1),r}}(g||(g={}));var _,b=Object(p.connect)((function(e){return{digitPanel:e.digitPanel}}),(function(e,t){return{targetPanel:function(t,n){e.setState({digitPanel:n})}}})),j=Object(p.connect)((function(e){return{panel:e.panel}}),(function(e,t){return{pickPanel:function(t,n){e.setState({panel:n})}}})),y=Object(p.connect)((function(e){return{session:e.session,random:e.random,results:e.results,penalty:e.penalty,finished:e.finished}}),(function(e,t){return{newSession:function(t){return e.setState({session:Math.random(),penalty:0,finished:!1,random:g.shuffleGenerator(r.IMAGES.length)})},updateResult:function(t,n,r){var a=t.results;a[n]=r,e.setState({results:a})},updateScore:function(t,n){var r=t.penalty;e.setState({penalty:r+n})},submit:function(t,n){var r=t.penalty;e.setState({penalty:r+n,finished:!0})}}})),O=n(0),x=y(b((function(e){e.finished;var t=e.session;e.digitPanel,e.targetPanel,e.updateScore;return fetch("https://api.dictionaryapi.dev/api/v2/entries/en_US/hello",{method:"GET",headers:{"Content-Type":"application/json"}}).then((function(e){return console.log({res:e})})),Object(O.jsx)("div",{},t)}))),S=n(5),D=n(4),E=function(e){var t=e.digit,n=e.draggable,r=e.onDragStart,a=e.onDragEnd;Object(D.a)(e,["digit","draggable","onDragStart","onDragEnd"]);return Object(O.jsx)("div",{draggable:n,onDragStart:r,onDragEnd:a,className:"digit",children:t})},w=y((function(e){var t=e.results,n=e.penalty,r=e.newSession,a=e.finished,o=e.submit;return Object(O.jsxs)("div",{style:{position:"fixed",top:200,right:0,width:180,paddingRight:40,textAlign:"center",zIndex:999},children:[Object(O.jsx)("div",{style:{fontWeight:"bold",margin:16,cursor:"pointer",height:96,backgroundImage:"url(".concat("/math","/reset.png)"),backgroundSize:"contain",backgroundRepeat:"no-repeat",backgroundPosition:"center"},onClick:r}),Object(O.jsx)("div",{style:{fontFamily:"SaucerBB",fontSize:80,color:a?"red":"grey"},children:100+n}),Object(O.jsx)("div",{style:{fontWeight:"bold",backgroundColor:"yellow",padding:4,border:"2px solid blue",cursor:"pointer",pointerEvents:a?"none":"auto"},onClick:function(){var e=t.reduce((function(e,n){return e+100/t.length*~~n}),0);o(Math.ceil(e)-100)},children:"SUBMIT"})]})})),P=n(8),I=y((function(e){var t=e.session,n=e.random,a=e.onDragOver,o=e.onDrop,c=(Object(D.a)(e,["session","random","onDragOver","onDrop"]),i.a.useMemo((function(){return r.IMAGES[n.next().value]}),[t]));return Object(O.jsx)("div",{onDragOver:a,onDrop:o,children:Object(O.jsx)("div",{className:"digit-panel",style:{backgroundImage:"url(".concat("/math","/images/").concat(c,")")}})})})),M=b((function(e){var t=e.value,n=e.mask,r=void 0===n?t:n,a=e.targetPanel,o=(Object(D.a)(e,["value","mask","targetPanel"]),i.a.useRef(null));return t=("x".repeat(r.length)+t).slice(-r.length),Object(O.jsx)("div",{ref:o,style:{display:"flex",justifyContent:"flex-end"},children:Object(m.a)(r).map((function(e,n){return isFinite(+e)?Object(O.jsx)(E,{digit:+e,draggable:!1},n):Object(O.jsx)(I,{onDragOver:function(e){return e.preventDefault()},onDrop:function(e){return a({validDigit:t[n],panel:e.currentTarget})}},n)}))})})),k=y((function(e){var t,n=e.id,r=e.operands,a=e.finished,o=e.updateResult,c=(Object(D.a)(e,["id","operands","finished","updateResult"]),i.a.useRef(null)),s=i.a.useRef(null),u=i.a.useRef(null),l=Object(S.a)(r,2),d=l[0],h=l[1],p=i.a.useMemo((function(){var e=d>h?"-":"+";return{operator:e,result:{"+":d+h,"-":d-h}[e]}}),[]),g=p.operator,f=p.result,_=!!u.current&&+(null===(t=u.current)||void 0===t?void 0:t.innerText.replace(/\s/g,""))===f;return i.a.useEffect((function(){function e(){return(e=Object(P.a)(v.a.mark((function e(){return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:o(n,_);case 1:case"end":return e.stop()}}),e)})))).apply(this,arguments)}!function(){e.apply(this,arguments)}()})),Object(O.jsxs)("div",{className:"calculation",style:{background:a&&!_?"orchid":"transparent"},children:[Object(O.jsx)("div",{ref:c,children:Object(O.jsx)(M,{value:String(d)})}),Object(O.jsx)("div",{className:"operator",children:g}),Object(O.jsx)("div",{ref:s,children:Object(O.jsx)(M,{value:String(h)})}),Object(O.jsx)("div",{style:{width:"100%",height:10,background:"#0051f5",color:"transparent",margin:"16px 0px"},children:"=="}),Object(O.jsx)("div",{ref:u,children:Object(O.jsx)(M,{value:String(f),mask:"x".repeat(Math.ceil(Math.log10(Math.max.apply(Math,Object(m.a)(r))))+1)})})]})})),T=y(b((function(e){var t=i.a.useState(0),n=Object(S.a)(t,2),a=(n[0],n[1]),o=e.finished,c=e.session,s=e.digitPanel,u=e.targetPanel,l=e.updateScore,d=i.a.useMemo((function(){return Object(m.a)(Array(r.SIZE)).map((function(){return[g.random(r.MIN,r.MAX),g.random(r.MIN,r.MAX)]}))}),[c]);return Object(O.jsxs)("div",{children:[Object(O.jsx)("div",{style:{display:"flex",position:"fixed",backgroundColor:"transparent",zIndex:999},children:Object(m.a)(Array(10)).map((function(e,t){return Object(O.jsx)(E,{digit:t,draggable:!o,onDragStart:function(e){return u(null)},onDragEnd:function(e){if(s){var t=s.validDigit,n=s.panel;n.innerHTML="",n.appendChild(g.cloneDigit(e.target,t,l)),a(Math.random())}}},t)}))}),Object(O.jsx)(w,{}),Object(O.jsx)("div",{className:"content-panel",children:Object(O.jsx)("div",{style:{display:"flex",flexFlow:"wrap",backgroundColor:"linen"},children:d.map((function(e,t){return Object(O.jsx)(k,{id:String(t),operands:e},t)}))})})]},c)}))),C=function(e){var t=e.letter,n=e.draggable,r=e.onDragStart,a=e.onDragEnd;Object(D.a)(e,["letter","draggable","onDragStart","onDragEnd"]);return Object(O.jsx)("div",{draggable:n,onDragStart:r,onDragEnd:a,className:"letter",children:t})},A=y(j((function(e){var t=e.finished,n=e.panel,r=e.pickPanel,a=e.updateScore;return Object(O.jsx)("div",{style:{display:"flex",flexFlow:"wrap",position:"fixed",backgroundColor:"transparent",zIndex:999},children:Object(m.a)("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map((function(e){return Object(O.jsx)(C,{letter:e,draggable:!t,onDragStart:function(e){return r(null)},onDragEnd:function(e){n&&(n.innerHTML="",n.appendChild(g.cloneLetter(e.target,n.valid,a)))}},e)}))})}))),H=y((function(e){var t=e.session,n=e.random,a=e.onDragOver,o=e.onDrop,c=(Object(D.a)(e,["session","random","onDragOver","onDrop"]),i.a.useMemo((function(){return r.IMAGES[n.next().value]}),[t]));return Object(O.jsx)("div",{onDragOver:a,onDrop:o,children:Object(O.jsx)("div",{className:"letter-panel",style:{backgroundImage:"url(".concat("/math","/images/").concat(c,")")}})})})),L=function(e){var t=e.word,n=i.a.useMemo((function(){var e="".concat(t).concat("_".repeat(t.length)),n=document.createElement("audio"),r=document.createElement("source");return r.src="https://www.oxfordlearnersdictionaries.com/media/american_english/us_pron/"+"/".concat(e.slice(0,1))+"/".concat(e.slice(0,3))+"/".concat(e.slice(0,5))+"/".concat(t,"__us_1.mp3"),r.type="audio/mpeg",n.appendChild(r),n}),[t]);return Object(O.jsx)("div",{style:{width:180,textAlign:"center",zIndex:999},children:Object(O.jsx)("div",{style:{fontWeight:"bold",margin:16,cursor:"pointer",height:96,backgroundImage:"url(".concat("/math","/speaker.png)"),backgroundSize:"contain",backgroundRepeat:"no-repeat",backgroundPosition:"center"},onClick:function(){return n.play()}})})},R=function(e){var t=i.a.useState(!1),n=Object(S.a)(t,2),r=n[0],a=n[1],o=e.word,c=i.a.useMemo((function(){function e(){return(e=Object(P.a)(v.a.mark((function e(){var t,n,r,a;return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t=[],n=function(){var e=Object(P.a)(v.a.mark((function e(t){return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.abrupt("return",new Promise((function(e){var n=new FileReader;n.onload=function(){var t=n.result,r=null===t||void 0===t?void 0:t.split(","),a=Object(S.a)(r,2),o=(a[0],a[1]);e(o)},n.readAsDataURL(t)})));case 1:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}(),e.next=4,navigator.mediaDevices.getUserMedia({audio:!0});case 4:return r=e.sent,(a=new MediaRecorder(r)).ondataavailable=function(){var e=Object(P.a)(v.a.mark((function e(r){var o,i,c;return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(t.push(r.data),"inactive"!==a.state){e.next=16;break}return e.prev=2,o=new Blob(t,{type:"audio/webm"}),e.next=6,n(o);case 6:return i=e.sent,e.next=9,fetch("https://speech.googleapis.com/v1/speech:recognize",{method:"POST",headers:{"Content-Type":"application/json",key:"AIzaSyAWEriX0ahrPBMHozN9zCfqxuyCwxVBFhs"},body:JSON.stringify({audio:{content:i},config:{enableAutomaticPunctuation:!1,encoding:"LINEAR16",languageCode:"en-US",sampleRateHertz:16e3,maxAlternatives:30}})});case 9:c=e.sent,console.log({res:c}),e.next=16;break;case 13:e.prev=13,e.t0=e.catch(2),console.log(e.t0);case 16:case"end":return e.stop()}}),e,null,[[2,13]])})));return function(t){return e.apply(this,arguments)}}(),a.onstop=function(){return t.length=0},e.abrupt("return",a);case 9:case"end":return e.stop()}}),e)})))).apply(this,arguments)}document.createElement("audio");return function(){return e.apply(this,arguments)}()}),[o]);return Object(O.jsx)("div",{style:{width:180,textAlign:"center",zIndex:999},children:Object(O.jsx)("div",{className:"button",style:{backgroundImage:"url(".concat("/math","/").concat(r?"recording":"recorder",".png)")},onClick:function(){return c.then((function(e){a(e.start()||!0),setTimeout((function(){return a(e.stop()||!1)}),2e3)}))}})})},N=y(j((function(e){var t=e.word,n=e.pickPanel;return Object(O.jsxs)("div",{className:"word",children:[Object(O.jsxs)("div",{style:{display:"flex"},children:[Object(O.jsx)("div",{className:"word-picture",style:{backgroundImage:"url(".concat("/math","/wild-animals/").concat(t,"-150x150.png)")}}),Object(O.jsxs)("div",{style:{display:"flex",flexDirection:"column"},children:[Object(O.jsx)(L,{word:t}),Object(O.jsx)(R,{word:t})]})]}),Object(O.jsx)("div",{className:"word-panel",children:Object(m.a)(t.toUpperCase()).map((function(e,t){return Object(O.jsx)(H,{onDragOver:function(e){return e.preventDefault()},onDrop:function(t){var r=t.currentTarget;r.valid=e,n(r)}},t)}))})]})}))),z=y(b((function(e){var t=i.a.useState(0),n=Object(S.a)(t,2),a=(n[0],n[1],e.finished,e.session),o=(e.digitPanel,e.targetPanel,e.updateScore,i.a.useMemo((function(){return r.WILD_ANIMALS[g.random(0,r.WILD_ANIMALS.length)]}),[a]));return Object(O.jsxs)("div",{children:[Object(O.jsx)(A,{}),Object(O.jsx)(w,{}),Object(O.jsx)(N,{word:o})]},a)})));n(23),n(24);!function(e){var t;!function(e){e[e.WELCOME=0]="WELCOME",e[e.MATH=1]="MATH",e[e.ENGLISH=2]="ENGLISH"}(t||(t={})),e.ETypes=t;var n=[],a=h()({currentPage:t.MATH,results:[],penalty:0,finished:!1,random:g.shuffleGenerator(r.IMAGES.length)}),o=Object(p.connect)((function(e){return{currentPage:e.currentPage}}),(function(e,t){return{push:function(t,r){var a=t.currentPage;n.push(a),e.setState({currentPage:r})},pop:function(t){var r=n.pop();e.setState({currentPage:r})}}}))((function(e){return function(e){var n;return n={},Object(u.a)(n,t.WELCOME,Object(O.jsx)(x,Object(l.a)({},e),"PageWelcome")),Object(u.a)(n,t.MATH,Object(O.jsx)(T,Object(l.a)({},e),"PageMath")),Object(u.a)(n,t.ENGLISH,Object(O.jsx)(z,Object(l.a)({},e),"PageEnglish")),n}(e)[e.currentPage]}));e.Display=function(){return Object(O.jsx)(p.Provider,{store:a,children:Object(O.jsx)(o,{})})}}(_||(_={}));n(25);var F=function(){return Object(O.jsx)(_.Display,{})};Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));var W=function(e){e&&e instanceof Function&&n.e(3).then(n.bind(null,27)).then((function(t){var n=t.getCLS,r=t.getFID,a=t.getFCP,o=t.getLCP,i=t.getTTFB;n(e),r(e),a(e),o(e),i(e)}))};s.a.render(Object(O.jsx)(F,{}),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()})).catch((function(e){console.error(e.message)})),W()}},[[26,1,2]]]);
//# sourceMappingURL=main.64dd37ca.chunk.js.map