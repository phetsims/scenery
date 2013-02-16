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
    phet.assert( trail instanceof scenery.Trail, 'trail is not a trail' );
    this.trail = trail;
    
    this.setBefore( isBefore );
  };
  var TrailPointer = scenery.TrailPointer;
  
  TrailPointer.prototype = {
    constructor: TrailPointer,
    
    copy: function() {
      return new TrailPointer( this.trail.copy(), this.isBefore );
    },
    
    setBefore: function( isBefore ) {
      this.isBefore = isBefore;
      this.isAfter = !isBefore;
    },
    
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
      phet.assert( other );
      
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
    
    // TODO: refactor with "Side"-like handling
    // moves this pointer forwards one step in the nested order
    nestedForwards: function() {
      if ( this.isBefore ) {
        if ( this.trail.lastNode().children.length > 0 ) {
          // stay as before, just walk to the first child
          this.trail.addDescendant( this.trail.lastNode().children[0], 0 );
        } else {
          // stay on the same node, but switch to after
          this.setBefore( false );
        }
      } else {
        if ( this.trail.indices.length === 0 ) {
          // nothing else to jump to below, so indicate the lack of existence
          return null;
        } else {
          var index = this.trail.indices[this.trail.indices.length - 1];
          this.trail.removeDescendant();
          
          if ( this.trail.lastNode().children.length > index + 1 ) {
            // more siblings, switch to the beginning of the next one
            this.trail.addDescendant( this.trail.lastNode().children[index+1], index + 1 );
            this.setBefore( true );
          } else {
            // no more siblings. exit on parent. nothing else needed since we're already isAfter
          }
        }
      }
    },
    
    // moves this pointer backwards one step in the nested order
    nestedBackwards: function() {
      if ( this.isBefore ) {
        if ( this.trail.indices.length === 0 ) {
          // jumping off the front
          return null;
        } else {
          var index = this.trail.indices[this.trail.indices.length - 1];
          this.trail.removeDescendant();
          
          if ( index - 1 >= 0 ) {
            // more siblings, switch to the beginning of the previous one and switch to isAfter
            this.trail.addDescendant( this.trail.lastNode().children[index-1], index - 1 );
            this.setBefore( false );
          } else {
            // no more siblings. enter on parent. nothing else needed since we're already isBefore
          }
        }
      } else {
        if ( this.trail.lastNode().children.length > 0 ) {
          // stay isAfter, but walk to the last child
          var children = this.trail.lastNode().children;
          this.trail.addDescendant( children[children.length-1], children.length - 1 );
        } else {
          // switch to isBefore, since this is a leaf node
          this.setBefore( true );
        }
      }
    },
    
    // treats the pointer as render-ordered
    eachNodeBetween: function( other, callback ) {
      phet.assert( this.compareRender( other ) === -1, 'TrailPointer.eachNodeBetween pointers out of order, possibly in both meanings of the phrase!' );
      phet.assert( this.trail[0] === other.trail[0], 'TrailPointer.eachNodeBetween takes pointers with the same root' );
      
      // sanity check TODO: remove later
      this.trail.reindex();
      other.trail.reindex();
      
      throw new Error( 'eachNodeBetween unimplemented' );
    },
    
    // TODO: consider rename to depthFirstBetween
    /*
     * Recursively (depth-first) iterates over all pointers between this pointer and 'other', calling
     * callback( pointer ) for each pointer. If excludeEndpoints is truthy, the callback will not be
     * called if pointer is equivalent to this pointer or 'other'.
     *
     * If the callback returns a truthy value, the subtree for the current pointer will be skipped
     * (applies only to before-pointers)
     */
    eachPointerBetween: function( other, callback, excludeEndpoints ) {
      // make sure this pointer is before the other
      phet.assert( this.compareNested( other ) === -1, 'TrailPointer.eachBetween pointers out of order, possibly in both meanings of the phrase!' );
      phet.assert( this.trail[0] === other.trail[0], 'TrailPointer.eachBetween takes pointers with the same root' );
      
      // sanity check TODO: remove later
      this.trail.reindex();
      other.trail.reindex();
      
      var pointer = this.copy();
      
      var first = true;
      
      while ( !pointer.equalsNested( other ) ) {
        var skipSubtree = false;
        
        if ( first ) {
          // start point
          if ( !excludeEndpoints ) {
            skipSubtree = callback( pointer );
          }
          first = false;
        } else {
          // between point
          skipSubtree = callback( pointer );
        }
        
        if ( skipSubtree && pointer.isBefore ) {
          // to skip the subtree, we just change to isAfter
          pointer.setBefore( false );
        } else {
          pointer.nestedForwards();
        }
      }
      
      // end point
      if ( !excludeEndpoints ) {
        callback( pointer );
      }
    }
  };
  
})();

