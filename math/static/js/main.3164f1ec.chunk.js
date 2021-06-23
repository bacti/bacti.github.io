(this["webpackJsonpkha-app"]=this["webpackJsonpkha-app"]||[]).push([[0],{16:function(e,t,n){var r=n(38),a=r,o=Function("return this")();a.exportSymbol("proto.AssetRequest",null,o),a.exportSymbol("proto.AssetResponse",null,o),proto.AssetRequest=function(e){r.Message.initialize(this,e,0,-1,null,null)},a.inherits(proto.AssetRequest,r.Message),a.DEBUG&&!COMPILED&&(proto.AssetRequest.displayName="proto.AssetRequest"),r.Message.GENERATE_TO_OBJECT&&(proto.AssetRequest.prototype.toObject=function(e){return proto.AssetRequest.toObject(e,this)},proto.AssetRequest.toObject=function(e,t){var n={name:t.getName()};return e&&(n.$jspbMessageInstance=t),n}),proto.AssetRequest.deserializeBinary=function(e){var t=new r.BinaryReader(e),n=new proto.AssetRequest;return proto.AssetRequest.deserializeBinaryFromReader(n,t)},proto.AssetRequest.deserializeBinaryFromReader=function(e,t){for(;t.nextField()&&!t.isEndGroup();){switch(t.getFieldNumber()){case 1:var n=t.readString();e.setName(n);break;default:t.skipField()}}return e},proto.AssetRequest.serializeBinaryToWriter=function(e,t){e.serializeBinaryToWriter(t)},proto.AssetRequest.prototype.serializeBinary=function(){var e=new r.BinaryWriter;return this.serializeBinaryToWriter(e),e.getResultBuffer()},proto.AssetRequest.prototype.serializeBinaryToWriter=function(e){var t;(t=this.getName()).length>0&&e.writeString(1,t)},proto.AssetRequest.prototype.cloneMessage=function(){return r.Message.cloneMessage(this)},proto.AssetRequest.prototype.getName=function(){return r.Message.getFieldProto3(this,1,"")},proto.AssetRequest.prototype.setName=function(e){r.Message.setField(this,1,e)},proto.AssetResponse=function(e){r.Message.initialize(this,e,0,-1,null,null)},a.inherits(proto.AssetResponse,r.Message),a.DEBUG&&!COMPILED&&(proto.AssetResponse.displayName="proto.AssetResponse"),r.Message.GENERATE_TO_OBJECT&&(proto.AssetResponse.prototype.toObject=function(e){return proto.AssetResponse.toObject(e,this)},proto.AssetResponse.toObject=function(e,t){var n={data:t.getData_asB64()};return e&&(n.$jspbMessageInstance=t),n}),proto.AssetResponse.deserializeBinary=function(e){var t=new r.BinaryReader(e),n=new proto.AssetResponse;return proto.AssetResponse.deserializeBinaryFromReader(n,t)},proto.AssetResponse.deserializeBinaryFromReader=function(e,t){for(;t.nextField()&&!t.isEndGroup();){switch(t.getFieldNumber()){case 1:var n=t.readBytes();e.setData(n);break;default:t.skipField()}}return e},proto.AssetResponse.serializeBinaryToWriter=function(e,t){e.serializeBinaryToWriter(t)},proto.AssetResponse.prototype.serializeBinary=function(){var e=new r.BinaryWriter;return this.serializeBinaryToWriter(e),e.getResultBuffer()},proto.AssetResponse.prototype.serializeBinaryToWriter=function(e){var t;(t=this.getData_asU8()).length>0&&e.writeBytes(1,t)},proto.AssetResponse.prototype.cloneMessage=function(){return r.Message.cloneMessage(this)},proto.AssetResponse.prototype.getData=function(){return r.Message.getFieldProto3(this,1,"")},proto.AssetResponse.prototype.getData_asB64=function(){return r.Message.bytesAsB64(this.getData())},proto.AssetResponse.prototype.getData_asU8=function(){return r.Message.bytesAsU8(this.getData())},proto.AssetResponse.prototype.setData=function(e){r.Message.setField(this,1,e)},a.object.extend(t,proto)},23:function(e,t,n){var r=n(16),a=n(15).grpc,o=function(){function e(){}return e.serviceName="asset",e}();function s(e,t){this.serviceHost=e,this.options=t||{}}o.fetch={methodName:"fetch",service:o,requestStream:!1,responseStream:!1,requestType:r.AssetRequest,responseType:r.AssetResponse},t.asset=o,s.prototype.fetch=function(e,t,n){2===arguments.length&&(n=arguments[1]);var r=a.unary(o.fetch,{request:e,host:this.serviceHost,metadata:t,transport:this.options.transport,debug:this.options.debug,onEnd:function(e){if(n)if(e.status!==a.Code.OK){var t=new Error(e.statusMessage);t.code=e.status,t.metadata=e.trailers,n(t,null)}else n(null,e.message)}});return{cancel:function(){n=null,r.close()}}},t.assetClient=s},30:function(e,t,n){},31:function(e,t){var n;!function(e){"use strict";var t=function(){function e(){this._dropEffect="move",this._effectAllowed="all",this._data={}}return Object.defineProperty(e.prototype,"dropEffect",{get:function(){return this._dropEffect},set:function(e){this._dropEffect=e},enumerable:!0,configurable:!0}),Object.defineProperty(e.prototype,"effectAllowed",{get:function(){return this._effectAllowed},set:function(e){this._effectAllowed=e},enumerable:!0,configurable:!0}),Object.defineProperty(e.prototype,"types",{get:function(){return Object.keys(this._data)},enumerable:!0,configurable:!0}),e.prototype.clearData=function(e){null!=e?delete this._data[e]:this._data=null},e.prototype.getData=function(e){return this._data[e]||""},e.prototype.setData=function(e,t){this._data[e]=t},e.prototype.setDragImage=function(e,t,r){var a=n._instance;a._imgCustom=e,a._imgOffset={x:t,y:r}},e}();e.DataTransfer=t;var n=function(){function e(){if(this._lastClick=0,e._instance)throw"DragDropTouch instance already created.";var t=!1;if(document.addEventListener("test",(function(){}),{get passive(){return t=!0,!0}}),"ontouchstart"in document){var n=document,r=this._touchstart.bind(this),a=this._touchmove.bind(this),o=this._touchend.bind(this),s=!!t&&{passive:!1,capture:!1};n.addEventListener("touchstart",r,s),n.addEventListener("touchmove",a,s),n.addEventListener("touchend",o),n.addEventListener("touchcancel",o)}}return e.getInstance=function(){return e._instance},e.prototype._touchstart=function(t){var n=this;if(this._shouldHandle(t)){if(Date.now()-this._lastClick<e._DBLCLICK&&this._dispatchEvent(t,"dblclick",t.target))return t.preventDefault(),void this._reset();this._reset();var r=this._closestDraggable(t.target);r&&(this._dispatchEvent(t,"mousemove",t.target)||this._dispatchEvent(t,"mousedown",t.target)||(this._dragSource=r,this._ptDown=this._getPoint(t),this._lastTouch=t,t.preventDefault(),setTimeout((function(){n._dragSource==r&&null==n._img&&n._dispatchEvent(t,"contextmenu",r)&&n._reset()}),e._CTXMENU),e._ISPRESSHOLDMODE&&(this._pressHoldInterval=setTimeout((function(){n._isDragEnabled=!0,n._touchmove(t)}),e._PRESSHOLDAWAIT))))}},e.prototype._touchmove=function(e){if(this._shouldCancelPressHoldMove(e))this._reset();else if(this._shouldHandleMove(e)||this._shouldHandlePressHoldMove(e)){var t=this._getTarget(e);if(this._dispatchEvent(e,"mousemove",t))return this._lastTouch=e,void e.preventDefault();this._dragSource&&!this._img&&this._shouldStartDragging(e)&&(this._dispatchEvent(e,"dragstart",this._dragSource),this._createImage(e),this._dispatchEvent(e,"dragenter",t)),this._img&&(this._lastTouch=e,e.preventDefault(),t!=this._lastTarget&&(this._dispatchEvent(this._lastTouch,"dragleave",this._lastTarget),this._dispatchEvent(e,"dragenter",t),this._lastTarget=t),this._moveImage(e),this._isDropZone=this._dispatchEvent(e,"dragover",t))}},e.prototype._touchend=function(e){if(this._shouldHandle(e)){if(this._dispatchEvent(this._lastTouch,"mouseup",e.target))return void e.preventDefault();this._img||(this._dragSource=null,this._dispatchEvent(this._lastTouch,"click",e.target),this._lastClick=Date.now()),this._destroyImage(),this._dragSource&&(e.type.indexOf("cancel")<0&&this._isDropZone&&this._dispatchEvent(this._lastTouch,"drop",this._lastTarget),this._dispatchEvent(this._lastTouch,"dragend",this._dragSource),this._reset())}},e.prototype._shouldHandle=function(e){return e&&!e.defaultPrevented&&e.touches&&e.touches.length<2},e.prototype._shouldHandleMove=function(t){return!e._ISPRESSHOLDMODE&&this._shouldHandle(t)},e.prototype._shouldHandlePressHoldMove=function(t){return e._ISPRESSHOLDMODE&&this._isDragEnabled&&t&&t.touches&&t.touches.length},e.prototype._shouldCancelPressHoldMove=function(t){return e._ISPRESSHOLDMODE&&!this._isDragEnabled&&this._getDelta(t)>e._PRESSHOLDMARGIN},e.prototype._shouldStartDragging=function(t){var n=this._getDelta(t);return n>e._THRESHOLD||e._ISPRESSHOLDMODE&&n>=e._PRESSHOLDTHRESHOLD},e.prototype._reset=function(){this._destroyImage(),this._dragSource=null,this._lastTouch=null,this._lastTarget=null,this._ptDown=null,this._isDragEnabled=!1,this._isDropZone=!1,this._dataTransfer=new t,clearInterval(this._pressHoldInterval)},e.prototype._getPoint=function(e,t){return e&&e.touches&&(e=e.touches[0]),{x:t?e.pageX:e.clientX,y:t?e.pageY:e.clientY}},e.prototype._getDelta=function(t){if(e._ISPRESSHOLDMODE&&!this._ptDown)return 0;var n=this._getPoint(t);return Math.abs(n.x-this._ptDown.x)+Math.abs(n.y-this._ptDown.y)},e.prototype._getTarget=function(e){for(var t=this._getPoint(e),n=document.elementFromPoint(t.x,t.y);n&&"none"==getComputedStyle(n).pointerEvents;)n=n.parentElement;return n},e.prototype._createImage=function(t){this._img&&this._destroyImage();var n=this._imgCustom||this._dragSource;if(this._img=n.cloneNode(!0),this._copyStyle(n,this._img),this._img.style.top=this._img.style.left="-9999px",!this._imgCustom){var r=n.getBoundingClientRect(),a=this._getPoint(t);this._imgOffset={x:a.x-r.left,y:a.y-r.top},this._img.style.opacity=e._OPACITY.toString()}this._moveImage(t),document.body.appendChild(this._img)},e.prototype._destroyImage=function(){this._img&&this._img.parentElement&&this._img.parentElement.removeChild(this._img),this._img=null,this._imgCustom=null},e.prototype._moveImage=function(e){var t=this;requestAnimationFrame((function(){if(t._img){var n=t._getPoint(e,!0),r=t._img.style;r.position="absolute",r.pointerEvents="none",r.zIndex="999999",r.left=Math.round(n.x-t._imgOffset.x)+"px",r.top=Math.round(n.y-t._imgOffset.y)+"px"}}))},e.prototype._copyProps=function(e,t,n){for(var r=0;r<n.length;r++){var a=n[r];e[a]=t[a]}},e.prototype._copyStyle=function(t,n){if(e._rmvAtts.forEach((function(e){n.removeAttribute(e)})),t instanceof HTMLCanvasElement){var r=t,a=n;a.width=r.width,a.height=r.height,a.getContext("2d").drawImage(r,0,0)}for(var o=getComputedStyle(t),s=0;s<o.length;s++){var i=o[s];i.indexOf("transition")<0&&(n.style[i]=o[i])}n.style.pointerEvents="none";for(s=0;s<t.children.length;s++)this._copyStyle(t.children[s],n.children[s])},e.prototype._dispatchEvent=function(t,n,r){if(t&&r){var a=document.createEvent("Event"),o=t.touches?t.touches[0]:t;return a.initEvent(n,!0,!0),a.button=0,a.which=a.buttons=1,this._copyProps(a,t,e._kbdProps),this._copyProps(a,o,e._ptProps),a.dataTransfer=this._dataTransfer,r.dispatchEvent(a),a.defaultPrevented}return!1},e.prototype._closestDraggable=function(e){for(;e;e=e.parentElement)if(e.hasAttribute("draggable")&&e.draggable)return e;return null},e}();n._instance=new n,n._THRESHOLD=5,n._OPACITY=.5,n._DBLCLICK=500,n._CTXMENU=900,n._ISPRESSHOLDMODE=!1,n._PRESSHOLDAWAIT=400,n._PRESSHOLDMARGIN=25,n._PRESSHOLDTHRESHOLD=0,n._rmvAtts="id,class,style,draggable".split(","),n._kbdProps="altKey,ctrlKey,metaKey,shiftKey".split(","),n._ptProps="pageX,pageY,clientX,clientY,screenX,screenY".split(","),e.DragDropTouch=n}(n||(n={}))},42:function(e,t,n){},43:function(e,t,n){},44:function(e,t,n){},45:function(e,t,n){"use strict";n.r(t);var r,a=n(1),o=n.n(a),s=n(20),i=n.n(s),c=(n(30),n(31),n(8)),u=n(7),l=n(21),p=n.n(l),d=n(9),g=["character-0.png","character-1.svg","character-2.svg","character-3.svg","character-4.svg","character-5.png","character-6.png","character-7.png","character-8.png","character-9.png","character-10.png","character-11.png","character-12.png","character-13.png","character-14.png","character-15.png","character-16.png","character-17.png","character-18.png","character-19.png","character-20.png","character-21.png","character-22.png","character-23.png","character-24.png","character-25.png","character-26.png","character-27.svg","character-28.png","character-29.png","character-30.png","character-31.png","character-32.png","character-33.png","character-34.png","character-35.png","character-36.png","tamako-nobi.png","tamako-nobi-2.png","nobisuke-nobi.png","edmond-elephant.png","suzy-sheep.png","chloe-pig.png","alexander-pig.png","candy-cat.png","danny-dog.png","rebecca-rabbit.png","richard-rabbit.png","miss-rabbit.png","zoe-zebra.png","pedro-horse.png","madame-gazelle.png","eiichirou-senjou.png","hodol-choe.png","jaiko-gouda.png","michiko-minamoto.png","mr-kaminari.png","mrs-gouda.png","mrs-honekawa.png","nanhyang-gim.png"],h=["alligator","antelope","bear","camel","chimpanzee","coyote","crab","crocodile","dolphin","elephant","giraffe","gorilla","jellyfish","koala","leopard","lion","monkey","owl","panda","raccoon","shark","starfishes","walrus","wolf","woodpecker","zebra","badger","bat","deer","elk","fox","frog","hare","hedgehog","kangaroo","lizard","mole","otter","rabbit","rat","reindeer","snake","squirrel","toad"],f=n(2),v=n.n(f),m=n(5);!function(e){var t=v.a.mark(n);e.trace=function(e){var t="".concat((window.performance.now()/1e3).toFixed(3),": ").concat(String(e),"\n");console.warn(t);var n=document.getElementById("log");n&&(n.innerText=t+n.innerText)},e.requestFullscreen=function(e){var t=e.requestFullscreen||e.webkitRequestFullscreen||e.mozRequestFullScreen||e.msRequestFullscreen;t&&t.apply(e)};function n(e){var n,r;return v.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:n=Object(m.a)(Array(e)).map((function(e,t){return t})),r=[];case 2:return 0===r.length&&(r=Object(m.a)(n).sort((function(){return Math.random()-.5}))),t.next=6,r.pop();case 6:t.next=2;break;case 8:case"end":return t.stop()}}),t)}e.shuffleGenerator=n;e.random=function(e,t){return~~(Math.random()*(t-e))+e},e.cloneDigit=function(e,t,n){var r=e.cloneNode(!0),a=r.innerText===t;return r.style.background=a?"green":"red",r.ondragend=function(e){var t=document.createElement("div");t.className="digit-panel",e.target.replaceWith(t)},n(~~a-1),r},e.cloneLetter=function(e,t,n){var r=e.cloneNode(!0),a=r.innerText===t;return r.style.background=a?"green":"red",r.ondragend=function(e){var t=document.createElement("div");t.className="letter-panel",e.target.replaceWith(t)},n(~~a-1),r}}(r||(r={}));var b,y=n(4),j=Object(d.connect)((function(e){return{digitPanel:e.digitPanel}}),(function(e,t){return{targetPanel:function(t,n){e.setState({digitPanel:n})}}})),_=Object(d.connect)((function(e){return{panel:e.panel}}),(function(e,t){return{pickPanel:function(t,n){e.setState({panel:n})}}})),O=Object(d.connect)((function(e){return{session:e.session,random:e.random,results:e.results,penalty:e.penalty,timer:e.timer,finished:e.finished}}),(function(e,t){return{newSession:function(t){return e.setState({session:Math.random(),penalty:0,finished:!1,random:r.shuffleGenerator(g.length)})},updateTimer:function(t){return e.setState({timer:Date.now()})},updateResult:function(t,n,r){var a=t.results;a[n]=r,e.setState({results:a})},updateScore:function(t,n){var r=t.penalty;e.setState({penalty:r+n})},submit:function(t,n){var r=t.penalty;e.setState({penalty:r+n,finished:!0})}}})),x=n(3),E=n(22),D=n.n(E),w=n(15),S=n(23),R=n(16);!function(e){var t=window.URL||window.webkitURL,n={svg:"image/svg+xml"},r={};e.fetch=function(){var e=Object(y.a)(v.a.mark((function e(a){var o,s,i,c,u,l,p,d,g;return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(s=a.match(/\.(\w+)$/i)||[],i=Object(x.a)(s,2),i[0],c=i[1],u=n[c],!r[a]){e.next=4;break}return e.abrupt("return",r[a]);case 4:return(l=new R.AssetRequest).setName(a),e.next=8,new Promise((function(e){return w.grpc.unary(S.asset.fetch,{request:l,host:"https://bacti.tk:8080",onEnd:function(t){var n=t.message,r=t.status;return e(r?null:n.getData())}})}));case 8:if(null!=(p=e.sent)){e.next=11;break}return e.abrupt("return",null);case 11:return e.next=13,D.a.loadAsync(p);case 13:return d=e.sent,e.next=16,null===(o=d.file(a))||void 0===o?void 0:o.async("arraybuffer");case 16:return g=e.sent,r[a]=t.createObjectURL(new Blob([g],{type:u})),e.abrupt("return",r[a]);case 19:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}()}(b||(b={}));var T,A=n(6),P=n(24),k=n.n(P),M=n(25),I=n.n(M);!function(e){var t;!function(e){e.IMPRESSION="IMPRESSION",e.ENGAGEMENT="ENGAGEMENT",e.SUBMIT="SUBMIT",e.REFRESH="REFRESH"}(t||(t={})),e.ETypes=t;var n=k.a.getInstance();n.init("e9d181173175fa90e6140967d58d18e0");var r=window.localStorage||{};r.anoId=r.anoId||I.a.v4(),n.setUserId("ano:".concat(r.anoId));e.logEvent=function(e){var t=e.event,r=Object(A.a)(e,["event"]),a=window.localStorage.getItem("username");n.logEvent(t,Object(u.a)(Object(u.a)({},r),{},{user:a}))}}(T||(T={}));var N,C=n(0),H=O(j((function(e){var t=e.push;return o.a.useMemo((function(){var e=Date.now();T.logEvent({event:T.ETypes.IMPRESSION}),Object(y.a)(v.a.mark((function n(){var r;return v.a.wrap((function(n){for(;;)switch(n.prev=n.next){case 0:r=[{family:"Digit-Font",source:"fonts/HP001_5_hang_bold.ttf"},{family:"Letter-Font",source:"fonts/FineCollege.ttf"},{family:"SaucerBB",source:"fonts/SaucerBB.ttf"}],Promise.all(r.map(function(){var e=Object(y.a)(v.a.mark((function e(t){var n,r,a;return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return n=t.family,r=t.source,e.t0=FontFace,e.t1=n,e.t2="url(",e.next=6,b.fetch(r);case 6:return e.t3=e.sent,e.t4=e.t2.concat.call(e.t2,e.t3,")"),a=new e.t0(e.t1,e.t4),e.next=11,a.load();case 11:document.fonts.add(a);case 12:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}())).then((function(){T.logEvent({event:T.ETypes.ENGAGEMENT,loading:Date.now()-e}),t(N.ETypes.PROFILE)})).catch((function(){return null}));case 2:case"end":return n.stop()}}),n)})))()}),[]),Object(C.jsx)("div",{style:{width:"100vw",height:"100vh",backgroundColor:"#000000"},children:Object(C.jsx)("div",{className:"loader"})})}))),L=function(e){var t=e.letter,n=e.draggable,r=e.onDragStart,a=e.onDragEnd;Object(A.a)(e,["letter","draggable","onDragStart","onDragEnd"]);return Object(C.jsx)("div",{draggable:n,onDragStart:r,onDragEnd:a,className:"letter",children:t})},B=O(_((function(e){var t=e.finished,n=e.panel,a=e.pickPanel,o=e.updateScore,s=e.onDragEnd,i=void 0===s?function(e){n&&(n.innerHTML="",n.appendChild(r.cloneLetter(e.target,n.valid,o)))}:s;return Object(C.jsx)("div",{style:{display:"flex",flexFlow:"wrap",position:"fixed",backgroundColor:"transparent",zIndex:999},children:Object(m.a)("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map((function(e){return Object(C.jsx)(L,{letter:e,draggable:!t,onDragStart:function(e){return a(null)},onDragEnd:i},e)}))})}))),F=O((function(e){var t=e.session,n=e.random,r=e.onDragOver,a=e.onDrop,s=(Object(A.a)(e,["session","random","onDragOver","onDrop"]),o.a.useState(null)),i=Object(x.a)(s,2),c=i[0],u=i[1];return o.a.useMemo((function(){Object(y.a)(v.a.mark((function e(){return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.t0=u,e.next=3,b.fetch("images/".concat(g[n.next().value]));case 3:return e.t1=e.sent,e.abrupt("return",(0,e.t0)(e.t1));case 5:case"end":return e.stop()}}),e)})))()}),[t]),Object(C.jsx)("div",{onDragOver:r,onDrop:a,children:Object(C.jsx)("div",{className:"letter-panel",style:{backgroundImage:"url(".concat(c,")")}})})})),z=_((function(e){var t=e.panel,n=e.push,r=e.pickPanel,a=o.a.useState("   "),s=Object(x.a)(a,2),i=s[0],c=s[1],u=function(e,t,n){var r=Object(m.a)(e);return r[n]=t,"  ".concat(r.join("").replace(/(\s+)/g," ").trim(),"  ")};return o.a.useEffect((function(){window.localStorage.getItem("username")&&n(N.ETypes.MATH)}),[]),Object(C.jsxs)("div",{children:[Object(C.jsx)(B,{onDragEnd:function(e){t&&c(u(i,e.target.innerText,t.index))}}),Object(C.jsxs)("div",{className:"word",children:[Object(C.jsx)("div",{className:"word-panel",style:{paddingTop:100},children:Object(m.a)(i).map((function(e,t){return" "==e?Object(C.jsx)(F,{onDragOver:function(e){return e.preventDefault()},onDrop:function(e){var n=e.currentTarget;n.index=t,r(n)}},t):Object(C.jsx)(L,{letter:e,draggable:!0,onDragEnd:function(e){c(u(i," ",t))}},t)}))}),Object(C.jsx)("div",{style:{fontSize:"x-large",fontWeight:"bold",backgroundColor:"yellow",margin:40,padding:"16px 53px",border:"4px solid blue",cursor:"pointer",pointerEvents:"auto"},onClick:function(){window.localStorage.setItem("username",i.toLowerCase().trim()),n(N.ETypes.MATH)},children:"SAVE"})]})]})})),q=function(e){var t=e.digit,n=e.draggable,r=e.onDragStart,a=e.onDragEnd;Object(A.a)(e,["digit","draggable","onDragStart","onDragEnd"]);return Object(C.jsx)("div",{draggable:n,onDragStart:r,onDragEnd:a,className:"digit",children:t})},G=O((function(e){var t=e.results,n=e.timer,r=e.penalty,a=e.newSession,s=e.finished,i=e.submit,c=o.a.useState(null),u=Object(x.a)(c,2),l=u[0],p=u[1],d=o.a.useMemo((function(){return Object(y.a)(v.a.mark((function e(){return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.t0=p,e.next=3,b.fetch("reset.png");case 3:return e.t1=e.sent,e.abrupt("return",(0,e.t0)(e.t1));case 5:case"end":return e.stop()}}),e)})))(),window.localStorage.getItem("username")}),[]),g=function(){return Math.ceil(t.reduce((function(e,n){return e+100/t.length*~~n}),0))};return Object(C.jsxs)("div",{style:{position:"fixed",top:200,right:0,width:180,paddingRight:40,textAlign:"center",zIndex:999},children:[Object(C.jsx)("div",{className:"user",children:d}),Object(C.jsx)("div",{style:{fontWeight:"bold",margin:16,cursor:"pointer",height:96,backgroundImage:"url(".concat(l,")"),backgroundSize:"contain",backgroundRepeat:"no-repeat",backgroundPosition:"center"},onClick:function(){T.logEvent({event:T.ETypes.REFRESH,score:g(),playing:Date.now()-n}),a()}}),Object(C.jsx)("div",{style:{fontFamily:"SaucerBB",fontSize:80,color:s?"red":"grey"},children:100+r}),Object(C.jsx)("div",{style:{fontWeight:"bold",backgroundColor:"yellow",padding:4,border:"2px solid blue",cursor:"pointer",pointerEvents:s?"none":"auto"},onClick:function(){var e=g();T.logEvent({event:T.ETypes.SUBMIT,score:e+r,playing:Date.now()-n}),i(e-100)},children:"SUBMIT"})]})})),W=O((function(e){var t=e.session,n=e.random,r=e.onDragOver,a=e.onDrop,s=(Object(A.a)(e,["session","random","onDragOver","onDrop"]),o.a.useState(null)),i=Object(x.a)(s,2),c=i[0],u=i[1];return o.a.useMemo((function(){Object(y.a)(v.a.mark((function e(){return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.t0=u,e.next=3,b.fetch("images/".concat(g[n.next().value]));case 3:return e.t1=e.sent,e.abrupt("return",(0,e.t0)(e.t1));case 5:case"end":return e.stop()}}),e)})))()}),[t]),Object(C.jsx)("div",{onDragOver:r,onDrop:a,children:Object(C.jsx)("div",{className:"digit-panel",style:{backgroundImage:"url(".concat(c,")")}})})})),U=j((function(e){var t=e.value,n=e.mask,r=void 0===n?t:n,a=e.targetPanel,s=(Object(A.a)(e,["value","mask","targetPanel"]),o.a.useRef(null));return t=("x".repeat(r.length)+t).slice(-r.length),Object(C.jsx)("div",{ref:s,style:{display:"flex",justifyContent:"flex-end"},children:Object(m.a)(r).map((function(e,n){return isFinite(+e)?Object(C.jsx)(q,{digit:+e,draggable:!1},n):Object(C.jsx)(W,{onDragOver:function(e){return e.preventDefault()},onDrop:function(e){return a({validDigit:t[n],panel:e.currentTarget})}},n)}))})})),X=O((function(e){var t,n=e.id,r=e.operands,a=e.finished,s=e.updateResult,i=(Object(A.a)(e,["id","operands","finished","updateResult"]),o.a.useRef(null)),c=o.a.useRef(null),u=o.a.useRef(null),l=Object(x.a)(r,2),p=l[0],d=l[1],g=o.a.useMemo((function(){var e=p>d?"-":"+";return{operator:e,result:{"+":p+d,"-":p-d}[e]}}),[]),h=g.operator,f=g.result,b=!!u.current&&+(null===(t=u.current)||void 0===t?void 0:t.innerText.replace(/\s/g,""))===f;return o.a.useEffect((function(){function e(){return(e=Object(y.a)(v.a.mark((function e(){return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:s(n,b);case 1:case"end":return e.stop()}}),e)})))).apply(this,arguments)}!function(){e.apply(this,arguments)}()})),Object(C.jsxs)("div",{className:"calculation",style:{background:a&&!b?"orchid":"transparent"},children:[Object(C.jsx)("div",{ref:i,children:Object(C.jsx)(U,{value:String(p)})}),Object(C.jsx)("div",{className:"operator",children:h}),Object(C.jsx)("div",{ref:c,children:Object(C.jsx)(U,{value:String(d)})}),Object(C.jsx)("div",{style:{width:"100%",height:10,background:"#0051f5",color:"transparent",margin:"16px 0px"},children:"=="}),Object(C.jsx)("div",{ref:u,children:Object(C.jsx)(U,{value:String(f),mask:"x".repeat(Math.ceil(Math.log10(Math.max.apply(Math,Object(m.a)(r))))+1)})})]})})),K=O(j((function(e){var t=o.a.useState(0),n=Object(x.a)(t,2),a=(n[0],n[1]),s=e.finished,i=e.session,c=e.digitPanel,u=e.targetPanel,l=e.updateScore,p=e.updateTimer,d=o.a.useMemo((function(){return Object(m.a)(Array(20)).map((function(){return[r.random(10,100),r.random(10,100)]}))}),[i]);return o.a.useEffect((function(){p()}),[i]),Object(C.jsxs)("div",{children:[Object(C.jsx)("div",{style:{display:"flex",position:"fixed",backgroundColor:"transparent",zIndex:999},children:Object(m.a)(Array(10)).map((function(e,t){return Object(C.jsx)(q,{digit:t,draggable:!s,onDragStart:function(e){return u(null)},onDragEnd:function(e){if(c){var t=c.validDigit,n=c.panel;n.innerHTML="",n.appendChild(r.cloneDigit(e.target,t,l)),a(Math.random())}}},t)}))}),Object(C.jsx)(G,{}),Object(C.jsx)("div",{className:"content-panel",children:Object(C.jsx)("div",{style:{display:"flex",flexFlow:"wrap",backgroundColor:"linen"},children:d.map((function(e,t){return Object(C.jsx)(X,{id:String(t),operands:e},t)}))})})]},i)}))),Y=function(e){var t=e.word,n=o.a.useMemo((function(){var e="".concat(t).concat("_".repeat(t.length)),n=document.createElement("audio"),r=document.createElement("source");return r.src="https://www.oxfordlearnersdictionaries.com/media/american_english/us_pron/"+"/".concat(e.slice(0,1))+"/".concat(e.slice(0,3))+"/".concat(e.slice(0,5))+"/".concat(t,"__us_1.mp3"),r.type="audio/mpeg",n.appendChild(r),n}),[t]);return Object(C.jsx)("div",{style:{width:180,textAlign:"center",zIndex:999},children:Object(C.jsx)("div",{style:{fontWeight:"bold",margin:16,cursor:"pointer",height:96,backgroundImage:"url(".concat("/math","/speaker.png)"),backgroundSize:"contain",backgroundRepeat:"no-repeat",backgroundPosition:"center"},onClick:function(){return n.play()}})})},J=function(e){var t=o.a.useState(!1),n=Object(x.a)(t,2),r=n[0],a=n[1],s=e.word,i=o.a.useMemo((function(){function e(){return(e=Object(y.a)(v.a.mark((function e(){var t,n,r,a;return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t=[],n=function(){var e=Object(y.a)(v.a.mark((function e(t){return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.abrupt("return",new Promise((function(e){var n=new FileReader;n.onload=function(){var t=n.result,r=null===t||void 0===t?void 0:t.split(","),a=Object(x.a)(r,2),o=(a[0],a[1]);e(o)},n.readAsDataURL(t)})));case 1:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}(),e.next=4,navigator.mediaDevices.getUserMedia({audio:!0});case 4:return r=e.sent,(a=new MediaRecorder(r)).ondataavailable=function(){var e=Object(y.a)(v.a.mark((function e(r){var o,s,i;return v.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(t.push(r.data),"inactive"!==a.state){e.next=16;break}return e.prev=2,o=new Blob(t,{type:"audio/webm"}),e.next=6,n(o);case 6:return s=e.sent,e.next=9,fetch("https://speech.googleapis.com/v1/speech:recognize",{method:"POST",headers:{"Content-Type":"application/json",key:"AIzaSyAWEriX0ahrPBMHozN9zCfqxuyCwxVBFhs"},body:JSON.stringify({audio:{content:s},config:{enableAutomaticPunctuation:!1,encoding:"LINEAR16",languageCode:"en-US",sampleRateHertz:16e3,maxAlternatives:30}})});case 9:i=e.sent,console.log({res:i}),e.next=16;break;case 13:e.prev=13,e.t0=e.catch(2),console.log(e.t0);case 16:case"end":return e.stop()}}),e,null,[[2,13]])})));return function(t){return e.apply(this,arguments)}}(),a.onstop=function(){return t.length=0},e.abrupt("return",a);case 9:case"end":return e.stop()}}),e)})))).apply(this,arguments)}document.createElement("audio");return function(){return e.apply(this,arguments)}()}),[s]);return Object(C.jsx)("div",{style:{width:180,textAlign:"center",zIndex:999},children:Object(C.jsx)("div",{className:"button",style:{backgroundImage:"url(".concat("/math","/").concat(r?"recording":"recorder",".png)")},onClick:function(){return i.then((function(e){a(e.start()||!0),setTimeout((function(){return a(e.stop()||!1)}),2e3)}))}})})},Z=O(_((function(e){var t=e.word,n=e.pickPanel;return Object(C.jsxs)("div",{className:"word",children:[Object(C.jsxs)("div",{style:{display:"flex"},children:[Object(C.jsx)("div",{className:"word-picture",style:{backgroundImage:"url(".concat("/math","/wild-animals/").concat(t,"-150x150.png)")}}),Object(C.jsxs)("div",{style:{display:"flex",flexDirection:"column"},children:[Object(C.jsx)(Y,{word:t}),Object(C.jsx)(J,{word:t})]})]}),Object(C.jsx)("div",{className:"word-panel",children:Object(m.a)(t.toUpperCase()).map((function(e,t){return Object(C.jsx)(F,{onDragOver:function(e){return e.preventDefault()},onDrop:function(t){var r=t.currentTarget;r.valid=e,n(r)}},t)}))})]})}))),$=O(j((function(e){var t=o.a.useState(0),n=Object(x.a)(t,2),a=(n[0],n[1],e.finished,e.session),s=(e.digitPanel,e.targetPanel,e.updateScore,o.a.useMemo((function(){return h[r.random(0,h.length)]}),[a]));return Object(C.jsxs)("div",{children:[Object(C.jsx)(B,{}),Object(C.jsx)(G,{}),Object(C.jsx)(Z,{word:s})]},a)})));n(42),n(43);!function(e){var t;!function(e){e[e.LOADING=0]="LOADING",e[e.PROFILE=1]="PROFILE",e[e.MATH=2]="MATH",e[e.ENGLISH=3]="ENGLISH"}(t||(t={})),e.ETypes=t;var n=[],a=p()({currentPage:t.LOADING,results:[],penalty:0,finished:!1,random:r.shuffleGenerator(g.length)}),o=Object(d.connect)((function(e){return{currentPage:e.currentPage}}),(function(e,t){return{push:function(t,r){var a=t.currentPage;n.push(a),e.setState({currentPage:r})},pop:function(t){var r=n.pop();e.setState({currentPage:r})}}}))((function(e){return function(e){var n;return n={},Object(c.a)(n,t.LOADING,Object(C.jsx)(H,Object(u.a)({},e),"PageLoading")),Object(c.a)(n,t.PROFILE,Object(C.jsx)(z,Object(u.a)({},e),"PageProfile")),Object(c.a)(n,t.MATH,Object(C.jsx)(K,Object(u.a)({},e),"PageMath")),Object(c.a)(n,t.ENGLISH,Object(C.jsx)($,Object(u.a)({},e),"PageEnglish")),n}(e)[e.currentPage]}));e.Display=function(){return Object(C.jsx)(d.Provider,{store:a,children:Object(C.jsx)(o,{})})}}(N||(N={}));n(44);var V=function(){return Object(C.jsx)(N.Display,{})};Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));var Q=function(e){e&&e instanceof Function&&n.e(3).then(n.bind(null,46)).then((function(t){var n=t.getCLS,r=t.getFID,a=t.getFCP,o=t.getLCP,s=t.getTTFB;n(e),r(e),a(e),o(e),s(e)}))};i.a.render(Object(C.jsx)(V,{}),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()})).catch((function(e){console.error(e.message)})),Q()}},[[45,1,2]]]);
//# sourceMappingURL=main.3164f1ec.chunk.js.map