// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  "use strict";
  
  /*
   * isBefore: whether this points to before the node (and its children) have been rendered, or after
   */
  scenery.TrailPointer = function( trail, isBefore ) {
    this.trail = trail;
    
    this.isBefore = isBefore;
    this.isAfter = !isBefore;
  };
  var TrailPointer = scenery.TrailPointer;
  
  TrailPointer.prototype = {
    constructor: TrailPointer,
    
    // return the equivalent pointer that swaps before and after (may return null if it doesn't exist)
    getSwappedPointer: function() {
      var newTrail = this.isBefore ? this.trail.previous() : this.trail.next();
      
      if ( newTrail === null ) {
        return null;
      } else {
        return new TrailPointer( newTrail, !this.isBefore );
      }
    }
  };
  
})();

