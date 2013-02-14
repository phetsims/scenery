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
    },
    
    getBeforePointer: function() {
      return this.isBefore ? this : this.getSwappedPointer();
    },
    
    getAfterPointer: function() {
      return this.isAfter ? this : this.getSwappedPointer();
    },
    
    /* 
     * In the render order, will return 0 if the pointers are equivalent, -1 if this pointer is before the
     * other pointer, and 1 if this pointer is after the other pointer.
     */
    compare: function( other ) {
      phet.assert( other !== null );
      
      var a = this.getBeforePointer();
      var b = other.getBeforePointer();
      
      if ( a !== null && b !== null ) {
        // normal (non-degenerate) case
        return a.trail.compare( b.trail );
      } else {
        // null "before" point is equivalent to the "after" pointer on the last rendered node.
        if ( a === b ) {
          return 0; // uniqueness guarantees they were the same
        } else {
          return a === null ? 1 : -1;
        }
      }
    },
    
    eachNodeBetween: function( other, callback ) {
      var comparison = this.trail.compare( other.trail );
    }
  };
  
})();

