(this["webpackJsonpkha-app"]=this["webpackJsonpkha-app"]||[]).push([[0],{19:function(t,e,n){},20:function(t,e){var n;!function(t){"use strict";var e=function(){function t(){this._dropEffect="move",this._effectAllowed="all",this._data={}}return Object.defineProperty(t.prototype,"dropEffect",{get:function(){return this._dropEffect},set:function(t){this._dropEffect=t},enumerable:!0,configurable:!0}),Object.defineProperty(t.prototype,"effectAllowed",{get:function(){return this._effectAllowed},set:function(t){this._effectAllowed=t},enumerable:!0,configurable:!0}),Object.defineProperty(t.prototype,"types",{get:function(){return Object.keys(this._data)},enumerable:!0,configurable:!0}),t.prototype.clearData=function(t){null!=t?delete this._data[t]:this._data=null},t.prototype.getData=function(t){return this._data[t]||""},t.prototype.setData=function(t,e){this._data[t]=e},t.prototype.setDragImage=function(t,e,r){var a=n._instance;a._imgCustom=t,a._imgOffset={x:e,y:r}},t}();t.DataTransfer=e;var n=function(){function t(){if(this._lastClick=0,t._instance)throw"DragDropTouch instance already created.";var e=!1;if(document.addEventListener("test",(function(){}),{get passive(){return e=!0,!0}}),"ontouchstart"in document){var n=document,r=this._touchstart.bind(this),a=this._touchmove.bind(this),i=this._touchend.bind(this),o=!!e&&{passive:!1,capture:!1};n.addEventListener("touchstart",r,o),n.addEventListener("touchmove",a,o),n.addEventListener("touchend",i),n.addEventListener("touchcancel",i)}}return t.getInstance=function(){return t._instance},t.prototype._touchstart=function(e){var n=this;if(this._shouldHandle(e)){if(Date.now()-this._lastClick<t._DBLCLICK&&this._dispatchEvent(e,"dblclick",e.target))return e.preventDefault(),void this._reset();this._reset();var r=this._closestDraggable(e.target);r&&(this._dispatchEvent(e,"mousemove",e.target)||this._dispatchEvent(e,"mousedown",e.target)||(this._dragSource=r,this._ptDown=this._getPoint(e),this._lastTouch=e,e.preventDefault(),setTimeout((function(){n._dragSource==r&&null==n._img&&n._dispatchEvent(e,"contextmenu",r)&&n._reset()}),t._CTXMENU),t._ISPRESSHOLDMODE&&(this._pressHoldInterval=setTimeout((function(){n._isDragEnabled=!0,n._touchmove(e)}),t._PRESSHOLDAWAIT))))}},t.prototype._touchmove=function(t){if(this._shouldCancelPressHoldMove(t))this._reset();else if(this._shouldHandleMove(t)||this._shouldHandlePressHoldMove(t)){var e=this._getTarget(t);if(this._dispatchEvent(t,"mousemove",e))return this._lastTouch=t,void t.preventDefault();this._dragSource&&!this._img&&this._shouldStartDragging(t)&&(this._dispatchEvent(t,"dragstart",this._dragSource),this._createImage(t),this._dispatchEvent(t,"dragenter",e)),this._img&&(this._lastTouch=t,t.preventDefault(),e!=this._lastTarget&&(this._dispatchEvent(this._lastTouch,"dragleave",this._lastTarget),this._dispatchEvent(t,"dragenter",e),this._lastTarget=e),this._moveImage(t),this._isDropZone=this._dispatchEvent(t,"dragover",e))}},t.prototype._touchend=function(t){if(this._shouldHandle(t)){if(this._dispatchEvent(this._lastTouch,"mouseup",t.target))return void t.preventDefault();this._img||(this._dragSource=null,this._dispatchEvent(this._lastTouch,"click",t.target),this._lastClick=Date.now()),this._destroyImage(),this._dragSource&&(t.type.indexOf("cancel")<0&&this._isDropZone&&this._dispatchEvent(this._lastTouch,"drop",this._lastTarget),this._dispatchEvent(this._lastTouch,"dragend",this._dragSource),this._reset())}},t.prototype._shouldHandle=function(t){return t&&!t.defaultPrevented&&t.touches&&t.touches.length<2},t.prototype._shouldHandleMove=function(e){return!t._ISPRESSHOLDMODE&&this._shouldHandle(e)},t.prototype._shouldHandlePressHoldMove=function(e){return t._ISPRESSHOLDMODE&&this._isDragEnabled&&e&&e.touches&&e.touches.length},t.prototype._shouldCancelPressHoldMove=function(e){return t._ISPRESSHOLDMODE&&!this._isDragEnabled&&this._getDelta(e)>t._PRESSHOLDMARGIN},t.prototype._shouldStartDragging=function(e){var n=this._getDelta(e);return n>t._THRESHOLD||t._ISPRESSHOLDMODE&&n>=t._PRESSHOLDTHRESHOLD},t.prototype._reset=function(){this._destroyImage(),this._dragSource=null,this._lastTouch=null,this._lastTarget=null,this._ptDown=null,this._isDragEnabled=!1,this._isDropZone=!1,this._dataTransfer=new e,clearInterval(this._pressHoldInterval)},t.prototype._getPoint=function(t,e){return t&&t.touches&&(t=t.touches[0]),{x:e?t.pageX:t.clientX,y:e?t.pageY:t.clientY}},t.prototype._getDelta=function(e){if(t._ISPRESSHOLDMODE&&!this._ptDown)return 0;var n=this._getPoint(e);return Math.abs(n.x-this._ptDown.x)+Math.abs(n.y-this._ptDown.y)},t.prototype._getTarget=function(t){for(var e=this._getPoint(t),n=document.elementFromPoint(e.x,e.y);n&&"none"==getComputedStyle(n).pointerEvents;)n=n.parentElement;return n},t.prototype._createImage=function(e){this._img&&this._destroyImage();var n=this._imgCustom||this._dragSource;if(this._img=n.cloneNode(!0),this._copyStyle(n,this._img),this._img.style.top=this._img.style.left="-9999px",!this._imgCustom){var r=n.getBoundingClientRect(),a=this._getPoint(e);this._imgOffset={x:a.x-r.left,y:a.y-r.top},this._img.style.opacity=t._OPACITY.toString()}this._moveImage(e),document.body.appendChild(this._img)},t.prototype._destroyImage=function(){this._img&&this._img.parentElement&&this._img.parentElement.removeChild(this._img),this._img=null,this._imgCustom=null},t.prototype._moveImage=function(t){var e=this;requestAnimationFrame((function(){if(e._img){var n=e._getPoint(t,!0),r=e._img.style;r.position="absolute",r.pointerEvents="none",r.zIndex="999999",r.left=Math.round(n.x-e._imgOffset.x)+"px",r.top=Math.round(n.y-e._imgOffset.y)+"px"}}))},t.prototype._copyProps=function(t,e,n){for(var r=0;r<n.length;r++){var a=n[r];t[a]=e[a]}},t.prototype._copyStyle=function(e,n){if(t._rmvAtts.forEach((function(t){n.removeAttribute(t)})),e instanceof HTMLCanvasElement){var r=e,a=n;a.width=r.width,a.height=r.height,a.getContext("2d").drawImage(r,0,0)}for(var i=getComputedStyle(e),o=0;o<i.length;o++){var s=i[o];s.indexOf("transition")<0&&(n.style[s]=i[s])}n.style.pointerEvents="none";for(o=0;o<e.children.length;o++)this._copyStyle(e.children[o],n.children[o])},t.prototype._dispatchEvent=function(e,n,r){if(e&&r){var a=document.createEvent("Event"),i=e.touches?e.touches[0]:e;return a.initEvent(n,!0,!0),a.button=0,a.which=a.buttons=1,this._copyProps(a,e,t._kbdProps),this._copyProps(a,i,t._ptProps),a.dataTransfer=this._dataTransfer,r.dispatchEvent(a),a.defaultPrevented}return!1},t.prototype._closestDraggable=function(t){for(;t;t=t.parentElement)if(t.hasAttribute("draggable")&&t.draggable)return t;return null},t}();n._instance=new n,n._THRESHOLD=5,n._OPACITY=.5,n._DBLCLICK=500,n._CTXMENU=900,n._ISPRESSHOLDMODE=!1,n._PRESSHOLDAWAIT=400,n._PRESSHOLDMARGIN=25,n._PRESSHOLDTHRESHOLD=0,n._rmvAtts="id,class,style,draggable".split(","),n._kbdProps="altKey,ctrlKey,metaKey,shiftKey".split(","),n._ptProps="pageX,pageY,clientX,clientY,screenX,screenY".split(","),t.DragDropTouch=n}(n||(n={}))},23:function(t,e,n){},24:function(t,e,n){},25:function(t,e,n){},26:function(t,e,n){"use strict";n.r(e);var r,a,i=n(1),o=n.n(i),s=n(11),c=n.n(s),u=(n(19),n(20),n(8)),l=n(12),h=n(13),d=n.n(h),g=n(5);(a=r||(r={})).IMAGES=["character-0.png","character-1.svg","character-2.svg","character-3.svg","character-4.svg","character-5.png","character-6.png","character-7.png","character-8.jpg","character-9.jpg","character-10.png","character-11.png","character-12.jpg","character-13.jpg","character-14.png","character-15.png","character-16.png","character-17.png","character-18.png","character-19.png","character-20.png","character-21.png","character-22.png","character-23.png","character-24.png","character-25.png","character-26.png","character-27.svg","character-28.png"],a.SIZE=20,a.MIN=10,a.MAX=100;var p,f=n(6),_=n.n(f),v=n(3);!function(t){var e=_.a.mark(n);function n(t){var n,r;return _.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:n=Object(v.a)(Array(t)).map((function(t,e){return e})),r=[];case 2:return 0===r.length&&(r=Object(v.a)(n).sort((function(){return Math.random()-.5}))),e.next=6,r.pop();case 6:e.next=2;break;case 8:case"end":return e.stop()}}),e)}t.shuffleGenerator=n;t.random=function(t,e){return~~(Math.random()*(e-t))+t},t.cloneDigit=function(t,e,n){var r=t.cloneNode(!0),a=r.innerText===e;return r.style.background=a?"green":"red",r.ondragend=function(t){var e=document.createElement("div");e.className="digit-panel",t.target.replaceWith(e)},n(~~a-1),r}}(p||(p={}));var m,b=n(4),y=n(2),O=n(0),j=function(t){var e=t.digit,n=t.draggable,r=t.onDragStart,a=t.onDragEnd;Object(y.a)(t,["digit","draggable","onDragStart","onDragEnd"]);return Object(O.jsx)("div",{draggable:n,onDragStart:r,onDragEnd:a,className:"digit",style:{fontFamily:"FineCollege"},children:e})},D=Object(g.connect)((function(t){return{digitPanel:t.digitPanel}}),(function(t,e){return{targetPanel:function(e,n){t.setState({digitPanel:n})}}})),S=Object(g.connect)((function(t){return{session:t.session,random:t.random,results:t.results,score:t.score}}),(function(t,e){return{newSession:function(e){return t.setState({session:Math.random(),score:0,random:p.shuffleGenerator(r.IMAGES.length)})},updateResult:function(e,n,r){var a=e.results;a[n]=r,t.setState({results:a})},updateScore:function(e,n){var r=e.score;t.setState({score:r+n})}}})),E=S((function(t){var e=t.results,n=t.score,r=t.updateScore,a=t.newSession,i=(Object(y.a)(t,["results","score","updateScore","newSession"]),o.a.useState(!1)),s=Object(b.a)(i,2),c=s[0],u=s[1];return Object(O.jsxs)("div",{style:{position:"fixed",top:124,right:0,width:180,paddingRight:40,textAlign:"center",zIndex:999},children:[Object(O.jsx)("div",{style:{fontWeight:"bold",margin:16,cursor:"pointer",height:96,backgroundImage:"url(".concat("/math","/reset.png)"),backgroundSize:"contain",backgroundRepeat:"no-repeat",backgroundPosition:"center"},onClick:a}),Object(O.jsx)("div",{style:{fontFamily:"SaucerBB",fontSize:80,color:c?"red":"grey"},children:100+n}),Object(O.jsx)("div",{style:{fontWeight:"bold",backgroundColor:"yellow",padding:4,border:"2px solid blue",cursor:"pointer"},onClick:function(){if(!c){var t=e.reduce((function(t,n){return t+100/e.length*~~n}),0);r(Math.ceil(t)-100),u(!0)}},children:"SUBMIT"})]})})),x=n(14),P=S((function(t){var e=t.session,n=t.random,a=t.onDragOver,i=t.onDrop,s=(Object(y.a)(t,["session","random","onDragOver","onDrop"]),o.a.useMemo((function(){return r.IMAGES[n.next().value]}),[e]));return Object(O.jsx)("div",{onDragOver:a,onDrop:i,children:Object(O.jsx)("div",{className:"digit-panel",style:{backgroundImage:"url(".concat("/math","/images/").concat(s,")")}})})})),M=D((function(t){var e=t.value,n=t.mask,r=void 0===n?e:n,a=t.targetPanel,i=(Object(y.a)(t,["value","mask","targetPanel"]),o.a.useRef(null));return e=("x".repeat(r.length)+e).slice(-r.length),Object(O.jsx)("div",{ref:i,style:{display:"flex",justifyContent:"flex-end"},children:Object(v.a)(r).map((function(t,n){return isFinite(+t)?Object(O.jsx)(j,{digit:+t,draggable:!1},n):Object(O.jsx)(P,{onDragOver:function(t){return t.preventDefault()},onDrop:function(t){return a({validDigit:e[n],panel:t.currentTarget})}},n)}))})})),I=S((function(t){var e=t.id,n=t.operands,r=t.updateResult,a=(Object(y.a)(t,["id","operands","updateResult"]),o.a.useRef(null)),i=o.a.useRef(null),s=o.a.useRef(null),c=Object(b.a)(n,2),u=c[0],l=c[1],h=o.a.useMemo((function(){var t=u>l?"-":"+";return{operator:t,result:{"+":u+l,"-":u-l}[t]}}),[]),d=h.operator,g=h.result;return o.a.useEffect((function(){function t(){return(t=Object(x.a)(_.a.mark((function t(){var n;return _.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:s.current&&r(e,+(null===(n=s.current)||void 0===n?void 0:n.innerText.replace(/\s/g,""))===g);case 1:case"end":return t.stop()}}),t)})))).apply(this,arguments)}!function(){t.apply(this,arguments)}()})),Object(O.jsxs)("div",{className:"calculation",children:[Object(O.jsx)("div",{ref:a,children:Object(O.jsx)(M,{value:String(u)})}),Object(O.jsx)("div",{className:"operator",children:d}),Object(O.jsx)("div",{ref:i,children:Object(O.jsx)(M,{value:String(l)})}),Object(O.jsx)("div",{style:{width:"100%",height:10,background:"#0051f5",color:"transparent",margin:"16px 0px"},children:"=="}),Object(O.jsx)("div",{ref:s,children:Object(O.jsx)(M,{value:String(g),mask:"x".repeat(Math.ceil(Math.log10(Math.max.apply(Math,Object(v.a)(n))))+1)})})]})})),T=S(D((function(t){var e=o.a.useState(0),n=Object(b.a)(e,2),a=(n[0],n[1]),i=t.session,s=t.digitPanel,c=t.targetPanel,u=t.updateScore,l=o.a.useMemo((function(){return Object(v.a)(Array(r.SIZE)).map((function(){return[p.random(r.MIN,r.MAX),p.random(r.MIN,r.MAX)]}))}),[i]);return Object(O.jsxs)("div",{children:[Object(O.jsx)("div",{style:{display:"flex",position:"fixed",backgroundColor:"transparent",zIndex:999},children:Object(v.a)(Array(10)).map((function(t,e){return Object(O.jsx)(j,{digit:e,draggable:!0,onDragStart:function(t){return c(null)},onDragEnd:function(t){if(s){var e=s.validDigit,n=s.panel;n.innerHTML="",n.appendChild(p.cloneDigit(t.target,e,u)),a(Math.random())}}},e)}))}),Object(O.jsx)(E,{}),Object(O.jsx)("div",{style:{position:"absolute",top:124,overflowY:"scroll",width:"100%",height:900},children:Object(O.jsx)("div",{style:{display:"flex",flexFlow:"wrap",backgroundColor:"linen"},children:l.map((function(t,e){return Object(O.jsx)(I,{id:String(e),operands:t},e)}))})})]},i)})));n(23),n(24);!function(t){var e;!function(t){t[t.DEMO=0]="DEMO"}(e||(e={})),t.ETypes=e;var n=[],a=d()({currentPage:e.DEMO,results:[],score:0,random:p.shuffleGenerator(r.IMAGES.length)}),i=Object(g.connect)((function(t){return{currentPage:t.currentPage}}),(function(t,e){return{push:function(e,r){var a=e.currentPage;n.push(a),t.setState({currentPage:r})},pop:function(e){var r=n.pop();t.setState({currentPage:r})}}}))((function(t){return function(t){return Object(u.a)({},e.DEMO,Object(O.jsx)(T,Object(l.a)({},t),"PageDemo"))}(t)[t.currentPage]}));t.Display=function(){return Object(O.jsx)(g.Provider,{store:a,children:Object(O.jsx)(i,{})})}}(m||(m={}));n(25);var H=function(){return Object(O.jsx)(m.Display,{})},w=function(t){t&&t instanceof Function&&n.e(3).then(n.bind(null,27)).then((function(e){var n=e.getCLS,r=e.getFID,a=e.getFCP,i=e.getLCP,o=e.getTTFB;n(t),r(t),a(t),i(t),o(t)}))};c.a.render(Object(O.jsx)(o.a.StrictMode,{children:Object(O.jsx)(H,{})}),document.getElementById("root")),w()}},[[26,1,2]]]);
//# sourceMappingURL=main.65ead6b4.chunk.js.map