(this["webpackJsonpkha-app"]=this["webpackJsonpkha-app"]||[]).push([[0],{19:function(t,e,n){},20:function(t,e){var n;!function(t){"use strict";var e=function(){function t(){this._dropEffect="move",this._effectAllowed="all",this._data={}}return Object.defineProperty(t.prototype,"dropEffect",{get:function(){return this._dropEffect},set:function(t){this._dropEffect=t},enumerable:!0,configurable:!0}),Object.defineProperty(t.prototype,"effectAllowed",{get:function(){return this._effectAllowed},set:function(t){this._effectAllowed=t},enumerable:!0,configurable:!0}),Object.defineProperty(t.prototype,"types",{get:function(){return Object.keys(this._data)},enumerable:!0,configurable:!0}),t.prototype.clearData=function(t){null!=t?delete this._data[t]:this._data=null},t.prototype.getData=function(t){return this._data[t]||""},t.prototype.setData=function(t,e){this._data[t]=e},t.prototype.setDragImage=function(t,e,r){var i=n._instance;i._imgCustom=t,i._imgOffset={x:e,y:r}},t}();t.DataTransfer=e;var n=function(){function t(){if(this._lastClick=0,t._instance)throw"DragDropTouch instance already created.";var e=!1;if(document.addEventListener("test",(function(){}),{get passive(){return e=!0,!0}}),"ontouchstart"in document){var n=document,r=this._touchstart.bind(this),i=this._touchmove.bind(this),a=this._touchend.bind(this),o=!!e&&{passive:!1,capture:!1};n.addEventListener("touchstart",r,o),n.addEventListener("touchmove",i,o),n.addEventListener("touchend",a),n.addEventListener("touchcancel",a)}}return t.getInstance=function(){return t._instance},t.prototype._touchstart=function(e){var n=this;if(this._shouldHandle(e)){if(Date.now()-this._lastClick<t._DBLCLICK&&this._dispatchEvent(e,"dblclick",e.target))return e.preventDefault(),void this._reset();this._reset();var r=this._closestDraggable(e.target);r&&(this._dispatchEvent(e,"mousemove",e.target)||this._dispatchEvent(e,"mousedown",e.target)||(this._dragSource=r,this._ptDown=this._getPoint(e),this._lastTouch=e,e.preventDefault(),setTimeout((function(){n._dragSource==r&&null==n._img&&n._dispatchEvent(e,"contextmenu",r)&&n._reset()}),t._CTXMENU),t._ISPRESSHOLDMODE&&(this._pressHoldInterval=setTimeout((function(){n._isDragEnabled=!0,n._touchmove(e)}),t._PRESSHOLDAWAIT))))}},t.prototype._touchmove=function(t){if(this._shouldCancelPressHoldMove(t))this._reset();else if(this._shouldHandleMove(t)||this._shouldHandlePressHoldMove(t)){var e=this._getTarget(t);if(this._dispatchEvent(t,"mousemove",e))return this._lastTouch=t,void t.preventDefault();this._dragSource&&!this._img&&this._shouldStartDragging(t)&&(this._dispatchEvent(t,"dragstart",this._dragSource),this._createImage(t),this._dispatchEvent(t,"dragenter",e)),this._img&&(this._lastTouch=t,t.preventDefault(),e!=this._lastTarget&&(this._dispatchEvent(this._lastTouch,"dragleave",this._lastTarget),this._dispatchEvent(t,"dragenter",e),this._lastTarget=e),this._moveImage(t),this._isDropZone=this._dispatchEvent(t,"dragover",e))}},t.prototype._touchend=function(t){if(this._shouldHandle(t)){if(this._dispatchEvent(this._lastTouch,"mouseup",t.target))return void t.preventDefault();this._img||(this._dragSource=null,this._dispatchEvent(this._lastTouch,"click",t.target),this._lastClick=Date.now()),this._destroyImage(),this._dragSource&&(t.type.indexOf("cancel")<0&&this._isDropZone&&this._dispatchEvent(this._lastTouch,"drop",this._lastTarget),this._dispatchEvent(this._lastTouch,"dragend",this._dragSource),this._reset())}},t.prototype._shouldHandle=function(t){return t&&!t.defaultPrevented&&t.touches&&t.touches.length<2},t.prototype._shouldHandleMove=function(e){return!t._ISPRESSHOLDMODE&&this._shouldHandle(e)},t.prototype._shouldHandlePressHoldMove=function(e){return t._ISPRESSHOLDMODE&&this._isDragEnabled&&e&&e.touches&&e.touches.length},t.prototype._shouldCancelPressHoldMove=function(e){return t._ISPRESSHOLDMODE&&!this._isDragEnabled&&this._getDelta(e)>t._PRESSHOLDMARGIN},t.prototype._shouldStartDragging=function(e){var n=this._getDelta(e);return n>t._THRESHOLD||t._ISPRESSHOLDMODE&&n>=t._PRESSHOLDTHRESHOLD},t.prototype._reset=function(){this._destroyImage(),this._dragSource=null,this._lastTouch=null,this._lastTarget=null,this._ptDown=null,this._isDragEnabled=!1,this._isDropZone=!1,this._dataTransfer=new e,clearInterval(this._pressHoldInterval)},t.prototype._getPoint=function(t,e){return t&&t.touches&&(t=t.touches[0]),{x:e?t.pageX:t.clientX,y:e?t.pageY:t.clientY}},t.prototype._getDelta=function(e){if(t._ISPRESSHOLDMODE&&!this._ptDown)return 0;var n=this._getPoint(e);return Math.abs(n.x-this._ptDown.x)+Math.abs(n.y-this._ptDown.y)},t.prototype._getTarget=function(t){for(var e=this._getPoint(t),n=document.elementFromPoint(e.x,e.y);n&&"none"==getComputedStyle(n).pointerEvents;)n=n.parentElement;return n},t.prototype._createImage=function(e){this._img&&this._destroyImage();var n=this._imgCustom||this._dragSource;if(this._img=n.cloneNode(!0),this._copyStyle(n,this._img),this._img.style.top=this._img.style.left="-9999px",!this._imgCustom){var r=n.getBoundingClientRect(),i=this._getPoint(e);this._imgOffset={x:i.x-r.left,y:i.y-r.top},this._img.style.opacity=t._OPACITY.toString()}this._moveImage(e),document.body.appendChild(this._img)},t.prototype._destroyImage=function(){this._img&&this._img.parentElement&&this._img.parentElement.removeChild(this._img),this._img=null,this._imgCustom=null},t.prototype._moveImage=function(t){var e=this;requestAnimationFrame((function(){if(e._img){var n=e._getPoint(t,!0),r=e._img.style;r.position="absolute",r.pointerEvents="none",r.zIndex="999999",r.left=Math.round(n.x-e._imgOffset.x)+"px",r.top=Math.round(n.y-e._imgOffset.y)+"px"}}))},t.prototype._copyProps=function(t,e,n){for(var r=0;r<n.length;r++){var i=n[r];t[i]=e[i]}},t.prototype._copyStyle=function(e,n){if(t._rmvAtts.forEach((function(t){n.removeAttribute(t)})),e instanceof HTMLCanvasElement){var r=e,i=n;i.width=r.width,i.height=r.height,i.getContext("2d").drawImage(r,0,0)}for(var a=getComputedStyle(e),o=0;o<a.length;o++){var s=a[o];s.indexOf("transition")<0&&(n.style[s]=a[s])}n.style.pointerEvents="none";for(o=0;o<e.children.length;o++)this._copyStyle(e.children[o],n.children[o])},t.prototype._dispatchEvent=function(e,n,r){if(e&&r){var i=document.createEvent("Event"),a=e.touches?e.touches[0]:e;return i.initEvent(n,!0,!0),i.button=0,i.which=i.buttons=1,this._copyProps(i,e,t._kbdProps),this._copyProps(i,a,t._ptProps),i.dataTransfer=this._dataTransfer,r.dispatchEvent(i),i.defaultPrevented}return!1},t.prototype._closestDraggable=function(t){for(;t;t=t.parentElement)if(t.hasAttribute("draggable")&&t.draggable)return t;return null},t}();n._instance=new n,n._THRESHOLD=5,n._OPACITY=.5,n._DBLCLICK=500,n._CTXMENU=900,n._ISPRESSHOLDMODE=!1,n._PRESSHOLDAWAIT=400,n._PRESSHOLDMARGIN=25,n._PRESSHOLDTHRESHOLD=0,n._rmvAtts="id,class,style,draggable".split(","),n._kbdProps="altKey,ctrlKey,metaKey,shiftKey".split(","),n._ptProps="pageX,pageY,clientX,clientY,screenX,screenY".split(","),t.DragDropTouch=n}(n||(n={}))},23:function(t,e,n){},24:function(t,e,n){},25:function(t,e,n){},26:function(t,e,n){"use strict";n.r(e);var r,i,a=n(1),o=n.n(a),s=n(11),c=n.n(s),u=(n(19),n(20),n(7)),l=n(12),d=n(13),h=n.n(d),p=n(5),g=n(3),f=n(4),_=n(2),v=n(0),m=function(t){var e=t.digit,n=t.draggable,r=t.onDragStart,i=t.onDragEnd;Object(_.a)(t,["digit","draggable","onDragStart","onDragEnd"]);return Object(v.jsx)("div",{draggable:n,onDragStart:r,onDragEnd:i,className:"digit",style:{fontFamily:"FineCollege"},children:e})},b=Object(p.connect)((function(t){return{digitPanel:t.digitPanel}}),(function(t,e){return{targetPanel:function(e,n){t.setState({digitPanel:n})}}})),y=Object(p.connect)((function(t){return{results:t.results,score:t.score}}),(function(t,e){return{updateResult:function(e,n,r){var i=e.results;i[n]=r,t.setState({results:i})},updateScore:function(e,n){var r=e.score;t.setState({score:r+n})}}})),O=y((function(t){var e=t.results,n=t.score,r=t.updateScore,i=(Object(_.a)(t,["results","score","updateScore"]),o.a.useState(!1)),a=Object(f.a)(i,2),s=a[0],c=a[1];return Object(v.jsxs)("div",{style:{position:"fixed",top:124,right:0,paddingRight:40,zIndex:999},children:[Object(v.jsx)("div",{style:{fontFamily:"SaucerBB",fontSize:80,color:"red"},children:100+n}),Object(v.jsx)("div",{style:{textAlign:"center",fontWeight:"bold",backgroundColor:"yellow",padding:4,border:"2px solid blue",cursor:"pointer"},onClick:function(){if(!s){var t=e.reduce((function(t,n){return t+100/e.length*~~n}),0);r(t-100),c(!0)}},children:"SUBMIT"})]})})),D=n(10),j=n.n(D),E=n(14),S=function(t){var e=t.onDragOver,n=t.onDrop;Object(_.a)(t,["onDragOver","onDrop"]);return Object(v.jsx)("div",{onDragOver:e,onDrop:n,children:Object(v.jsx)("div",{className:"digit-panel"})})},x=b((function(t){var e=t.value,n=t.mask,r=void 0===n?e:n,i=t.targetPanel,a=(Object(_.a)(t,["value","mask","targetPanel"]),o.a.useRef(null));return e=("x".repeat(r.length)+e).slice(-r.length),Object(v.jsx)("div",{ref:a,style:{display:"flex",justifyContent:"flex-end"},children:Object(g.a)(r).map((function(t,n){return isFinite(+t)?Object(v.jsx)(m,{digit:+t,draggable:!1},n):Object(v.jsx)(S,{onDragOver:function(t){return t.preventDefault()},onDrop:function(t){return i({validDigit:e[n],panel:t.currentTarget})}},n)}))})})),P=y((function(t){var e=t.id,n=t.operands,r=t.updateResult,i=(Object(_.a)(t,["id","operands","updateResult"]),o.a.useRef(null)),a=o.a.useRef(null),s=o.a.useRef(null),c=Object(f.a)(n,2),u=c[0],l=c[1];return o.a.useEffect((function(){function t(){return(t=Object(E.a)(j.a.mark((function t(){var n;return j.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:s.current&&r(e,+(null===(n=s.current)||void 0===n?void 0:n.innerText.replace(/\s/g,""))===u+l);case 1:case"end":return t.stop()}}),t)})))).apply(this,arguments)}!function(){t.apply(this,arguments)}()})),Object(v.jsxs)("div",{className:"calculation",children:[Object(v.jsx)("div",{ref:i,children:Object(v.jsx)(x,{value:String(u)})}),Object(v.jsx)("div",{className:"operator",children:"+"}),Object(v.jsx)("div",{ref:a,children:Object(v.jsx)(x,{value:String(l)})}),Object(v.jsx)("div",{style:{width:"100%",height:10,background:"#0051f5",color:"transparent",margin:"16px 0px"},children:"=="}),Object(v.jsx)("div",{ref:s,children:Object(v.jsx)(x,{value:String(u+l),mask:"x".repeat(Math.ceil(Math.log10(Math.max.apply(Math,Object(g.a)(n))))+1)})})]})}));(i=r||(r={})).random=function(t,e){return~~(Math.random()*(e-t))+t},i.cloneDigit=function(t,e,n){var r=t.cloneNode(!0),i=r.innerText===e;return r.style.background=i?"green":"red",r.ondragend=function(t){var e=document.createElement("div");e.className="digit-panel",t.target.replaceWith(e)},n(~~i-1),r};var T,H=y(b((function(t){var e=o.a.useState(0),n=Object(f.a)(e,2),i=(n[0],n[1]),a=t.digitPanel,s=t.targetPanel,c=t.updateScore,u=o.a.useMemo((function(){return Object(g.a)(Array(20)).map((function(){return[r.random(10,100),r.random(10,100)]}))}),[]);return Object(v.jsxs)(v.Fragment,{children:[Object(v.jsx)("div",{style:{display:"flex",position:"fixed"},children:Object(g.a)(Array(10)).map((function(t,e){return Object(v.jsx)(m,{digit:e,draggable:!0,onDragStart:function(t){return s(null)},onDragEnd:function(t){if(a){var e=a.validDigit,n=a.panel;n.innerHTML="",n.appendChild(r.cloneDigit(t.target,e,c)),i(Math.random())}}},e)}))}),Object(v.jsx)(O,{}),Object(v.jsx)("div",{style:{position:"absolute",top:124,overflowY:"scroll",width:"100%",height:900},children:Object(v.jsx)("div",{style:{display:"flex",flexFlow:"wrap",backgroundColor:"linen"},children:u.map((function(t,e){return Object(v.jsx)(P,{id:String(e),operands:t},e)}))})})]})})));n(23),n(24);!function(t){var e;!function(t){t[t.DEMO=0]="DEMO"}(e||(e={})),t.ETypes=e;var n=[],r=h()({currentPage:e.DEMO,results:[],score:0}),i=Object(p.connect)((function(t){return{currentPage:t.currentPage}}),(function(t,e){return{push:function(e,r){var i=e.currentPage;n.push(i),t.setState({currentPage:r})},pop:function(e){var r=n.pop();t.setState({currentPage:r})}}}))((function(t){return function(t){return Object(u.a)({},e.DEMO,Object(v.jsx)(H,Object(l.a)({},t),"PageDemo"))}(t)[t.currentPage]}));t.Display=function(){return Object(v.jsx)(p.Provider,{store:r,children:Object(v.jsx)(i,{})})}}(T||(T={}));n(25);var M=function(){return Object(v.jsx)(T.Display,{})},I=function(t){t&&t instanceof Function&&n.e(3).then(n.bind(null,27)).then((function(e){var n=e.getCLS,r=e.getFID,i=e.getFCP,a=e.getLCP,o=e.getTTFB;n(t),r(t),i(t),a(t),o(t)}))};c.a.render(Object(v.jsx)(o.a.StrictMode,{children:Object(v.jsx)(M,{})}),document.getElementById("root")),I()}},[[26,1,2]]]);
//# sourceMappingURL=main.8011c512.chunk.js.map