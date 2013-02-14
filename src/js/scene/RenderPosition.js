// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.RenderPosition = function( trail, side ) {
    this.trail = trail;
    this.side = side;
  };
  var RenderPosition = scenery.RenderPosition;
  
  RenderPosition.BEFORE = 1;
  RenderPosition.AFTER = 2;
  
  RenderPosition.prototype = {
    constructor: RenderPosition

  };
  
})();

