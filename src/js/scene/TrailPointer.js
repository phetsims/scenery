// Copyright 2002-2012, University of Colorado

/**
 * Points to a specific node (with a trail), and whether it is conceptually before or after the node.
 *
 * There are two orderings:
 * - rendering order: the order that node selves would be rendered, matching the Trail implicit order
 * - nesting order:   the order in depth first with entering a node being "before" and exiting a node being "after"
 *
 * TODO: more seamless handling of the orders. or just exclusively use the nesting order
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

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
    getRenderSwappedPointer: function() {
      var newTrail = this.isBefore ? this.trail.previous() : this.trail.next();
      
      if ( newTrail === null ) {
        return null;
      } else {
        return new TrailPointer( newTrail, !this.isBefore );
      }
    },
    
    getRenderBeforePointer: function() {
      return this.isBefore ? this : this.getRenderSwappedPointer();
    },
    
    getRenderAfterPointer: function() {
      return this.isAfter ? this : this.getRenderSwappedPointer();
    },
    
    /*
     * In the render order, will return 0 if the pointers are equivalent, -1 if this pointer is before the
     * other pointer, and 1 if this pointer is after the other pointer.
     */
    compareRender: function( other ) {
      phet.assert( other !== null );
      
      var a = this.getRenderBeforePointer();
      var b = other.getRenderBeforePointer();
      
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
    
    /*
     * Like compareRender, but for the nested (depth-first) order
     *
     * TODO: optimization?
     */
    compareNested: function( other ) {
      phet.assert( other !== null );
      
      var comparison = this.trail.compare( other.trail );
      
      if ( comparison === 0 ) {
        // if trails are equal, just compare before/after
        if ( this.isBefore === other.isBefore ) {
          return 0;
        } else {
          return this.isBefore ? -1 : 1;
        }
      } else {
        // if one is an extension of the other, the shorter isBefore flag determines the order completely
        if ( this.trail.isExtensionOf( other.trail ) ) {
          return other.isBefore ? 1 : -1;
        } else if ( other.trail.isExtensionOf( this.trail ) ) {
          return this.isBefore ? -1 : 1;
        } else {
          // neither is a subtrail of the other, so a straight trail comparison should give the answer
          return comparison;
        }
      }
    },
    
    equalsRender: function( other ) {
      return this.compareRender( other ) === 0;
    },
    
    equalsNested: function( other ) {
      return this.compareNested( other ) === 0;
    },
    
    // treats the pointer as render-ordered
    eachNodeBetween: function( other, callback ) {
      
    },
    
    // treats the pointer as nesting-ordered
    eachBetween: function( other, callbacks, includeEndpoints ) {
      
    }
  };
  
})();

