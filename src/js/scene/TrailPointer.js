// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.TrailPointer = function( trail, side ) {
    this.trail = trail;
    this.side = side;
  };
  var TrailPointer = scenery.TrailPointer;
  
  TrailPointer.BEFORE = 1;
  TrailPointer.AFTER = 2;
  
  TrailPointer.prototype = {
    constructor: TrailPointer

  };
  
})();

