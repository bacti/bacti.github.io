(this["webpackJsonpkha-app"]=this["webpackJsonpkha-app"]||[]).push([[0],{19:function(t,e,n){},20:function(t,e){var n;!function(t){"use strict";var e=function(){function t(){this._dropEffect="move",this._effectAllowed="all",this._data={}}return Object.defineProperty(t.prototype,"dropEffect",{get:function(){return this._dropEffect},set:function(t){this._dropEffect=t},enumerable:!0,configurable:!0}),Object.defineProperty(t.prototype,"effectAllowed",{get:function(){return this._effectAllowed},set:function(t){this._effectAllowed=t},enumerable:!0,configurable:!0}),Object.defineProperty(t.prototype,"types",{get:function(){return Object.keys(this._data)},enumerable:!0,configurable:!0}),t.prototype.clearData=function(t){null!=t?delete this._data[t]:this._data=null},t.prototype.getData=function(t){return this._data[t]||""},t.prototype.setData=function(t,e){this._data[t]=e},t.prototype.setDragImage=function(t,e,r){var a=n._instance;a._imgCustom=t,a._imgOffset={x:e,y:r}},t}();t.DataTransfer=e;var n=function(){function t(){if(this._lastClick=0,t._instance)throw"DragDropTouch instance already created.";var e=!1;if(document.addEventListener("test",(function(){}),{get passive(){return e=!0,!0}}),"ontouchstart"in document){var n=document,r=this._touchstart.bind(this),a=this._touchmove.bind(this),o=this._touchend.bind(this),i=!!e&&{passive:!1,capture:!1};n.addEventListener("touchstart",r,i),n.addEventListener("touchmove",a,i),n.addEventListener("touchend",o),n.addEventListener("touchcancel",o)}}return t.getInstance=function(){return t._instance},t.prototype._touchstart=function(e){var n=this;if(this._shouldHandle(e)){if(Date.now()-this._lastClick<t._DBLCLICK&&this._dispatchEvent(e,"dblclick",e.target))return e.preventDefault(),void this._reset();this._reset();var r=this._closestDraggable(e.target);r&&(this._dispatchEvent(e,"mousemove",e.target)||this._dispatchEvent(e,"mousedown",e.target)||(this._dragSource=r,this._ptDown=this._getPoint(e),this._lastTouch=e,e.preventDefault(),setTimeout((function(){n._dragSource==r&&null==n._img&&n._dispatchEvent(e,"contextmenu",r)&&n._reset()}),t._CTXMENU),t._ISPRESSHOLDMODE&&(this._pressHoldInterval=setTimeout((function(){n._isDragEnabled=!0,n._touchmove(e)}),t._PRESSHOLDAWAIT))))}},t.prototype._touchmove=function(t){if(this._shouldCancelPressHoldMove(t))this._reset();else if(this._shouldHandleMove(t)||this._shouldHandlePressHoldMove(t)){var e=this._getTarget(t);if(this._dispatchEvent(t,"mousemove",e))return this._lastTouch=t,void t.preventDefault();this._dragSource&&!this._img&&this._shouldStartDragging(t)&&(this._dispatchEvent(t,"dragstart",this._dragSource),this._createImage(t),this._dispatchEvent(t,"dragenter",e)),this._img&&(this._lastTouch=t,t.preventDefault(),e!=this._lastTarget&&(this._dispatchEvent(this._lastTouch,"dragleave",this._lastTarget),this._dispatchEvent(t,"dragenter",e),this._lastTarget=e),this._moveImage(t),this._isDropZone=this._dispatchEvent(t,"dragover",e))}},t.prototype._touchend=function(t){if(this._shouldHandle(t)){if(this._dispatchEvent(this._lastTouch,"mouseup",t.target))return void t.preventDefault();this._img||(this._dragSource=null,this._dispatchEvent(this._lastTouch,"click",t.target),this._lastClick=Date.now()),this._destroyImage(),this._dragSource&&(t.type.indexOf("cancel")<0&&this._isDropZone&&this._dispatchEvent(this._lastTouch,"drop",this._lastTarget),this._dispatchEvent(this._lastTouch,"dragend",this._dragSource),this._reset())}},t.prototype._shouldHandle=function(t){return t&&!t.defaultPrevented&&t.touches&&t.touches.length<2},t.prototype._shouldHandleMove=function(e){return!t._ISPRESSHOLDMODE&&this._shouldHandle(e)},t.prototype._shouldHandlePressHoldMove=function(e){return t._ISPRESSHOLDMODE&&this._isDragEnabled&&e&&e.touches&&e.touches.length},t.prototype._shouldCancelPressHoldMove=function(e){return t._ISPRESSHOLDMODE&&!this._isDragEnabled&&this._getDelta(e)>t._PRESSHOLDMARGIN},t.prototype._shouldStartDragging=function(e){var n=this._getDelta(e);return n>t._THRESHOLD||t._ISPRESSHOLDMODE&&n>=t._PRESSHOLDTHRESHOLD},t.prototype._reset=function(){this._destroyImage(),this._dragSource=null,this._lastTouch=null,this._lastTarget=null,this._ptDown=null,this._isDragEnabled=!1,this._isDropZone=!1,this._dataTransfer=new e,clearInterval(this._pressHoldInterval)},t.prototype._getPoint=function(t,e){return t&&t.touches&&(t=t.touches[0]),{x:e?t.pageX:t.clientX,y:e?t.pageY:t.clientY}},t.prototype._getDelta=function(e){if(t._ISPRESSHOLDMODE&&!this._ptDown)return 0;var n=this._getPoint(e);return Math.abs(n.x-this._ptDown.x)+Math.abs(n.y-this._ptDown.y)},t.prototype._getTarget=function(t){for(var e=this._getPoint(t),n=document.elementFromPoint(e.x,e.y);n&&"none"==getComputedStyle(n).pointerEvents;)n=n.parentElement;return n},t.prototype._createImage=function(e){this._img&&this._destroyImage();var n=this._imgCustom||this._dragSource;if(this._img=n.cloneNode(!0),this._copyStyle(n,this._img),this._img.style.top=this._img.style.left="-9999px",!this._imgCustom){var r=n.getBoundingClientRect(),a=this._getPoint(e);this._imgOffset={x:a.x-r.left,y:a.y-r.top},this._img.style.opacity=t._OPACITY.toString()}this._moveImage(e),document.body.appendChild(this._img)},t.prototype._destroyImage=function(){this._img&&this._img.parentElement&&this._img.parentElement.removeChild(this._img),this._img=null,this._imgCustom=null},t.prototype._moveImage=function(t){var e=this;requestAnimationFrame((function(){if(e._img){var n=e._getPoint(t,!0),r=e._img.style;r.position="absolute",r.pointerEvents="none",r.zIndex="999999",r.left=Math.round(n.x-e._imgOffset.x)+"px",r.top=Math.round(n.y-e._imgOffset.y)+"px"}}))},t.prototype._copyProps=function(t,e,n){for(var r=0;r<n.length;r++){var a=n[r];t[a]=e[a]}},t.prototype._copyStyle=function(e,n){if(t._rmvAtts.forEach((function(t){n.removeAttribute(t)})),e instanceof HTMLCanvasElement){var r=e,a=n;a.width=r.width,a.height=r.height,a.getContext("2d").drawImage(r,0,0)}for(var o=getComputedStyle(e),i=0;i<o.length;i++){var c=o[i];c.indexOf("transition")<0&&(n.style[c]=o[c])}n.style.pointerEvents="none";for(i=0;i<e.children.length;i++)this._copyStyle(e.children[i],n.children[i])},t.prototype._dispatchEvent=function(e,n,r){if(e&&r){var a=document.createEvent("Event"),o=e.touches?e.touches[0]:e;return a.initEvent(n,!0,!0),a.button=0,a.which=a.buttons=1,this._copyProps(a,e,t._kbdProps),this._copyProps(a,o,t._ptProps),a.dataTransfer=this._dataTransfer,r.dispatchEvent(a),a.defaultPrevented}return!1},t.prototype._closestDraggable=function(t){for(;t;t=t.parentElement)if(t.hasAttribute("draggable")&&t.draggable)return t;return null},t}();n._instance=new n,n._THRESHOLD=5,n._OPACITY=.5,n._DBLCLICK=500,n._CTXMENU=900,n._ISPRESSHOLDMODE=!1,n._PRESSHOLDAWAIT=400,n._PRESSHOLDMARGIN=25,n._PRESSHOLDTHRESHOLD=0,n._rmvAtts="id,class,style,draggable".split(","),n._kbdProps="altKey,ctrlKey,metaKey,shiftKey".split(","),n._ptProps="pageX,pageY,clientX,clientY,screenX,screenY".split(","),t.DragDropTouch=n}(n||(n={}))},23:function(t,e,n){},24:function(t,e,n){},25:function(t,e,n){},26:function(t,e,n){"use strict";n.r(e);var r,a,o=n(1),i=n.n(o),c=n(13),s=n.n(c),u=(n(19),n(20),n(8)),l=n(12),d=n(14),h=n.n(d),p=n(6);(a=r||(r={})).IMAGES=["character-0.png","character-1.svg","character-2.svg","character-3.svg","character-4.svg","character-5.png","character-6.png","character-7.png","character-8.jpg","character-9.png","character-10.png","character-11.png","character-12.png","character-13.png","character-14.png","character-15.png","character-16.png","character-17.png","character-18.png","character-19.png","character-20.png","character-21.png","character-22.png","character-23.png","character-24.png","character-25.png","character-26.png","character-27.svg","character-28.png","character-29.png","character-30.png","character-31.png","character-32.png","character-33.png","character-34.png"],a.WILD_ANIMALS=["alligator","antelope","bear","camel","chimpanzee","coyote","crab","crocodile","dolphin","elephant","giraffe","gorilla","jellyfish","koala","leopard","lion","monkey","owl","panda","raccoon","shark","starfishes","walrus","wolf","woodpecker","zebra","badger","bat","deer","elk","fox","frog","hare","hedgehog","kangaroo","lizard","mole","otter","rabbit","rat","reindeer","snake","squirrel","toad"],a.SIZE=20,a.MIN=10,a.MAX=100;var g,f=n(3),v=n.n(f),m=n(2);!function(t){var e=v.a.mark(n);t.trace=function(t){var e="".concat((window.performance.now()/1e3).toFixed(3),": ").concat(String(t),"\n");console.warn(e);var n=document.getElementById("log");n&&(n.innerText=e+n.innerText)},t.requestFullscreen=function(t){var e=t.requestFullscreen||t.webkitRequestFullscreen||t.mozRequestFullScreen||t.msRequestFullscreen;e&&e.apply(t)};function n(t){var n,r;return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:n=Object(m.a)(Array(t)).map((function(t,e){return e})),r=[];case 2:return 0===r.length&&(r=Object(m.a)(n).sort((function(){return Math.random()-.5}))),e.next=6,r.pop();case 6:e.next=2;break;case 8:case"end":return e.stop()}}),e)}t.shuffleGenerator=n;t.random=function(t,e){return~~(Math.random()*(e-t))+t},t.cloneDigit=function(t,e,n){var r=t.cloneNode(!0),a=r.innerText===e;return r.style.background=a?"green":"red",r.ondragend=function(t){var e=document.createElement("div");e.className="digit-panel",t.target.replaceWith(e)},n(~~a-1),r},t.cloneLetter=function(t,e,n){var r=t.cloneNode(!0),a=r.innerText===e;return r.style.background=a?"green":"red",r.ondragend=function(t){var e=document.createElement("div");e.className="letter-panel",t.target.replaceWith(e)},n(~~a-1),r}}(g||(g={}));var _,b=n(5),j=n(4),y=n(0),O=function(t){var e=t.digit,n=t.draggable,r=t.onDragStart,a=t.onDragEnd;Object(j.a)(t,["digit","draggable","onDragStart","onDragEnd"]);return Object(y.jsx)("div",{draggable:n,onDragStart:r,onDragEnd:a,className:"digit",style:{fontFamily:"FineCollege"},children:e})},x=Object(p.connect)((function(t){return{digitPanel:t.digitPanel}}),(function(t,e){return{targetPanel:function(e,n){t.setState({digitPanel:n})}}})),D=Object(p.connect)((function(t){return{panel:t.panel}}),(function(t,e){return{pickPanel:function(e,n){t.setState({panel:n})}}})),S=Object(p.connect)((function(t){return{session:t.session,random:t.random,results:t.results,penalty:t.penalty,finished:t.finished}}),(function(t,e){return{newSession:function(e){return t.setState({session:Math.random(),penalty:0,finished:!1,random:g.shuffleGenerator(r.IMAGES.length)})},updateResult:function(e,n,r){var a=e.results;a[n]=r,t.setState({results:a})},updateScore:function(e,n){var r=e.penalty;t.setState({penalty:r+n})},submit:function(e,n){var r=e.penalty;t.setState({penalty:r+n,finished:!0})}}})),E=S((function(t){var e=t.results,n=t.penalty,r=t.newSession,a=t.finished,o=t.submit;return Object(y.jsxs)("div",{style:{position:"fixed",top:200,right:0,width:180,paddingRight:40,textAlign:"center",zIndex:999},children:[Object(y.jsx)("div",{style:{fontWeight:"bold",margin:16,cursor:"pointer",height:96,backgroundImage:"url(".concat("/math","/reset.png)"),backgroundSize:"contain",backgroundRepeat:"no-repeat",backgroundPosition:"center"},onClick:r}),Object(y.jsx)("div",{style:{fontFamily:"SaucerBB",fontSize:80,color:a?"red":"grey"},children:100+n}),Object(y.jsx)("div",{style:{fontWeight:"bold",backgroundColor:"yellow",padding:4,border:"2px solid blue",cursor:"pointer",pointerEvents:a?"none":"auto"},onClick:function(){var t=e.reduce((function(t,n){return t+100/e.length*~~n}),0);o(Math.ceil(t)-100)},children:"SUBMIT"})]})})),w=n(7),P=S((function(t){var e=t.session,n=t.random,a=t.onDragOver,o=t.onDrop,c=(Object(j.a)(t,["session","random","onDragOver","onDrop"]),i.a.useMemo((function(){return r.IMAGES[n.next().value]}),[e]));return Object(y.jsx)("div",{onDragOver:a,onDrop:o,children:Object(y.jsx)("div",{className:"digit-panel",style:{backgroundImage:"url(".concat("/math","/images/").concat(c,")")}})})})),I=x((function(t){var e=t.value,n=t.mask,r=void 0===n?e:n,a=t.targetPanel,o=(Object(j.a)(t,["value","mask","targetPanel"]),i.a.useRef(null));return e=("x".repeat(r.length)+e).slice(-r.length),Object(y.jsx)("div",{ref:o,style:{display:"flex",justifyContent:"flex-end"},children:Object(m.a)(r).map((function(t,n){return isFinite(+t)?Object(y.jsx)(O,{digit:+t,draggable:!1},n):Object(y.jsx)(P,{onDragOver:function(t){return t.preventDefault()},onDrop:function(t){return a({validDigit:e[n],panel:t.currentTarget})}},n)}))})})),M=S((function(t){var e,n=t.id,r=t.operands,a=t.finished,o=t.updateResult,c=(Object(j.a)(t,["id","operands","finished","updateResult"]),i.a.useRef(null)),s=i.a.useRef(null),u=i.a.useRef(null),l=Object(b.a)(r,2),d=l[0],h=l[1],p=i.a.useMemo((function(){var t=d>h?"-":"+";return{operator:t,result:{"+":d+h,"-":d-h}[t]}}),[]),g=p.operator,f=p.result,_=!u.current||+(null===(e=u.current)||void 0===e?void 0:e.innerText.replace(/\s/g,""))===f;return i.a.useEffect((function(){function t(){return(t=Object(w.a)(v.a.mark((function t(){return v.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:o(n,_);case 1:case"end":return t.stop()}}),t)})))).apply(this,arguments)}!function(){t.apply(this,arguments)}()})),Object(y.jsxs)("div",{className:"calculation",style:{background:a&&!_?"orchid":"transparent"},children:[Object(y.jsx)("div",{ref:c,children:Object(y.jsx)(I,{value:String(d)})}),Object(y.jsx)("div",{className:"operator",children:g}),Object(y.jsx)("div",{ref:s,children:Object(y.jsx)(I,{value:String(h)})}),Object(y.jsx)("div",{style:{width:"100%",height:10,background:"#0051f5",color:"transparent",margin:"16px 0px"},children:"=="}),Object(y.jsx)("div",{ref:u,children:Object(y.jsx)(I,{value:String(f),mask:"x".repeat(Math.ceil(Math.log10(Math.max.apply(Math,Object(m.a)(r))))+1)})})]})})),k=S(x((function(t){var e=i.a.useState(0),n=Object(b.a)(e,2),a=(n[0],n[1]),o=t.finished,c=t.session,s=t.digitPanel,u=t.targetPanel,l=t.updateScore,d=i.a.useMemo((function(){return Object(m.a)(Array(r.SIZE)).map((function(){return[g.random(r.MIN,r.MAX),g.random(r.MIN,r.MAX)]}))}),[c]);return Object(y.jsxs)("div",{children:[Object(y.jsx)("div",{style:{display:"flex",position:"fixed",backgroundColor:"transparent",zIndex:999},children:Object(m.a)(Array(10)).map((function(t,e){return Object(y.jsx)(O,{digit:e,draggable:!o,onDragStart:function(t){return u(null)},onDragEnd:function(t){if(s){var e=s.validDigit,n=s.panel;n.innerHTML="",n.appendChild(g.cloneDigit(t.target,e,l)),a(Math.random())}}},e)}))}),Object(y.jsx)(E,{}),Object(y.jsx)("div",{className:"content-panel",children:Object(y.jsx)("div",{style:{display:"flex",flexFlow:"wrap",backgroundColor:"linen"},children:d.map((function(t,e){return Object(y.jsx)(M,{id:String(e),operands:t},e)}))})})]},c)}))),T=function(t){var e=t.letter,n=t.draggable,r=t.onDragStart,a=t.onDragEnd;Object(j.a)(t,["letter","draggable","onDragStart","onDragEnd"]);return Object(y.jsx)("div",{draggable:n,onDragStart:r,onDragEnd:a,className:"letter",children:e})},A=S(D((function(t){var e=t.finished,n=t.panel,r=t.pickPanel,a=t.updateScore;return Object(y.jsx)("div",{style:{display:"flex",flexFlow:"wrap",position:"fixed",backgroundColor:"transparent",zIndex:999},children:Object(m.a)("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map((function(t){return Object(y.jsx)(T,{letter:t,draggable:!e,onDragStart:function(t){return r(null)},onDragEnd:function(t){n&&(n.innerHTML="",n.appendChild(g.cloneLetter(t.target,n.valid,a)))}},t)}))})}))),H=S((function(t){var e=t.session,n=t.random,a=t.onDragOver,o=t.onDrop,c=(Object(j.a)(t,["session","random","onDragOver","onDrop"]),i.a.useMemo((function(){return r.IMAGES[n.next().value]}),[e]));return Object(y.jsx)("div",{onDragOver:a,onDrop:o,children:Object(y.jsx)("div",{className:"letter-panel",style:{backgroundImage:"url(".concat("/math","/images/").concat(c,")")}})})})),C=function(t){var e=t.word,n=i.a.useMemo((function(){var t="".concat(e).concat("_".repeat(e.length)),n=document.createElement("audio"),r=document.createElement("source");return r.src="https://www.oxfordlearnersdictionaries.com/media/american_english/us_pron/"+"/".concat(t.slice(0,1))+"/".concat(t.slice(0,3))+"/".concat(t.slice(0,5))+"/".concat(e,"__us_1.mp3"),r.type="audio/mpeg",n.appendChild(r),n}),[e]);return Object(y.jsx)("div",{style:{width:180,textAlign:"center",zIndex:999},children:Object(y.jsx)("div",{style:{fontWeight:"bold",margin:16,cursor:"pointer",height:96,backgroundImage:"url(".concat("/math","/speaker.png)"),backgroundSize:"contain",backgroundRepeat:"no-repeat",backgroundPosition:"center"},onClick:function(){return n.play()}})})},L=function(t){var e=i.a.useState(!1),n=Object(b.a)(e,2),r=n[0],a=n[1],o=t.word,c=i.a.useMemo((function(){function t(){return(t=Object(w.a)(v.a.mark((function t(){var e,n,r,a;return v.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return e=[],n=function(){var t=Object(w.a)(v.a.mark((function t(e){return v.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.abrupt("return",new Promise((function(t){var n=new FileReader;n.onload=function(){var e=n.result,r=null===e||void 0===e?void 0:e.split(","),a=Object(b.a)(r,2),o=(a[0],a[1]);t(o)},n.readAsDataURL(e)})));case 1:case"end":return t.stop()}}),t)})));return function(e){return t.apply(this,arguments)}}(),t.next=4,navigator.mediaDevices.getUserMedia({audio:!0});case 4:return r=t.sent,(a=new MediaRecorder(r)).ondataavailable=function(){var t=Object(w.a)(v.a.mark((function t(r){var o,i,c;return v.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:if(e.push(r.data),"inactive"!==a.state){t.next=16;break}return t.prev=2,o=new Blob(e,{type:"audio/webm"}),t.next=6,n(o);case 6:return i=t.sent,t.next=9,fetch("https://speech.googleapis.com/v1/speech:recognize",{method:"POST",headers:{"Content-Type":"application/json",key:"AIzaSyAWEriX0ahrPBMHozN9zCfqxuyCwxVBFhs"},body:JSON.stringify({audio:{content:i},config:{enableAutomaticPunctuation:!1,encoding:"LINEAR16",languageCode:"en-US",sampleRateHertz:16e3,maxAlternatives:30}})});case 9:c=t.sent,console.log({res:c}),t.next=16;break;case 13:t.prev=13,t.t0=t.catch(2),console.log(t.t0);case 16:case"end":return t.stop()}}),t,null,[[2,13]])})));return function(e){return t.apply(this,arguments)}}(),a.onstop=function(){return e.length=0},t.abrupt("return",a);case 9:case"end":return t.stop()}}),t)})))).apply(this,arguments)}document.createElement("audio");return function(){return t.apply(this,arguments)}()}),[o]);return Object(y.jsx)("div",{style:{width:180,textAlign:"center",zIndex:999},children:Object(y.jsx)("div",{className:"button",style:{backgroundImage:"url(".concat("/math","/").concat(r?"recording":"recorder",".png)")},onClick:function(){return c.then((function(t){a(t.start()||!0),setTimeout((function(){return a(t.stop()||!1)}),2e3)}))}})})},R=S(D((function(t){var e=t.word,n=t.pickPanel;return Object(y.jsxs)("div",{className:"word",children:[Object(y.jsxs)("div",{style:{display:"flex"},children:[Object(y.jsx)("div",{className:"word-picture",style:{backgroundImage:"url(".concat("/math","/wild-animals/").concat(e,"-150x150.png)")}}),Object(y.jsxs)("div",{style:{display:"flex",flexDirection:"column"},children:[Object(y.jsx)(C,{word:e}),Object(y.jsx)(L,{word:e})]})]}),Object(y.jsx)("div",{className:"word-panel",children:Object(m.a)(e.toUpperCase()).map((function(t,e){return Object(y.jsx)(H,{onDragOver:function(t){return t.preventDefault()},onDrop:function(e){var r=e.currentTarget;r.valid=t,n(r)}},e)}))})]})}))),N=S(x((function(t){var e=i.a.useState(0),n=Object(b.a)(e,2),a=(n[0],n[1],t.finished,t.session),o=(t.digitPanel,t.targetPanel,t.updateScore,i.a.useMemo((function(){return r.WILD_ANIMALS[g.random(0,r.WILD_ANIMALS.length)]}),[a]));return Object(y.jsxs)("div",{children:[Object(y.jsx)(A,{}),Object(y.jsx)(E,{}),Object(y.jsx)(R,{word:o})]},a)})));n(23),n(24);!function(t){var e;!function(t){t[t.MATH=0]="MATH",t[t.ENGLISH=1]="ENGLISH"}(e||(e={})),t.ETypes=e;var n=[],a=h()({currentPage:e.MATH,results:[],penalty:0,finished:!1,random:g.shuffleGenerator(r.IMAGES.length)}),o=Object(p.connect)((function(t){return{currentPage:t.currentPage}}),(function(t,e){return{push:function(e,r){var a=e.currentPage;n.push(a),t.setState({currentPage:r})},pop:function(e){var r=n.pop();t.setState({currentPage:r})}}}))((function(t){return function(t){var n;return n={},Object(u.a)(n,e.MATH,Object(y.jsx)(k,Object(l.a)({},t),"PageMath")),Object(u.a)(n,e.ENGLISH,Object(y.jsx)(N,Object(l.a)({},t),"PageEnglish")),n}(t)[t.currentPage]}));t.Display=function(){return Object(y.jsx)(p.Provider,{store:a,children:Object(y.jsx)(o,{})})}}(_||(_={}));n(25);var F=function(){return Object(y.jsx)(_.Display,{})};Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));var z=function(t){t&&t instanceof Function&&n.e(3).then(n.bind(null,27)).then((function(e){var n=e.getCLS,r=e.getFID,a=e.getFCP,o=e.getLCP,i=e.getTTFB;n(t),r(t),a(t),o(t),i(t)}))};s.a.render(Object(y.jsx)(F,{}),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(t){t.unregister()})).catch((function(t){console.error(t.message)})),z()}},[[26,1,2]]]);
//# sourceMappingURL=main.35cb4ee7.chunk.js.map