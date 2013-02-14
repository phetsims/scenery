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
    
    copy: function() {
      return new TrailPointer( this.trail.copy(), this.isBefore );
    },
    
    toggleBeforeAfter: function() {
      this.isBefore = !this.isBefore;
      this.isAfter = !this.isAfter;
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
    
    // moves this pointer forwards one step in the nested order
    nestedForwards: function() {
      if ( this.isBefore ) {
        if ( this.trail.lastNode().children.length > 0 ) {
          // stay as before, just walk to the first child
          this.trail.addDescendant( this.trail.lastNode().children[0], 0 );
        } else {
          // stay on the same node, but switch to after
          this.toggleBeforeAfter();
        }
      } else {
        throw new Error( 'unimplemented! finish Trail changes first' );
      }
    },
    
    // moves this pointer backwards one step in the nested order
    nestedBackwards: function() {
      
    },
    
    // treats the pointer as render-ordered
    eachNodeBetween: function( other, callback ) {
      
    },
    
    // TODO: try next/previous handling as an easier case!
    // TODO: get rid of this monstrosity!
    // treats the pointer as nesting-ordered. assumes that they are properly ordered
    eachBetween: function( other, callbacks, excludeEndpoints ) {
      // make sure this pointer is before the other
      phet.assert( this.compareNested( other ) === -1, 'TrailPointer.eachBetween pointers out of order, possibly in both meanings of the phrase!' );
      phet.assert( this.trail[0] === other.trail[0], 'TrailPointer.eachBetween takes pointers with the same root' );
      
      // for speed
      this.trail.reindex();
      other.trail.reindex();
      
      // our state
      var indexStack = [];
      var nodeStack = [ this.trail[0] ];
      var enterAfterNext = false;
      var exitAfterNext = false;
      var entered = false;
      var exited = false;
      var match = 0;
      
      function enter( node ) {
        if ( !exited ) {
          var depth = indexStack.length - 1;
          
          // don't fully match if other.isAfter
          if ( depth === match && ( other.isBefore || match + 1 !== other.trail.indices.length ) ) {
            if ( other.trail.indices[match] === indexStack[depth] ) {
              match++;
              
              // once we have matched
              if ( match === other.trail.indices.length ) {
                exited = true;
                if ( !excludeEndpoints ) {
                  callbacks.enter( node );
                }
                return;
              }
            }
          }
        }
        
        if ( entered && !exited ) {
          callbacks.enter( node );
        }
        
        // technically dead code, but left here in case we change things in the future
        // if ( !entered && enterAfterNext ) {
        //   entered = true;
        // }
        if ( !exited && exitAfterNext ) {
          exited = true;
        }
      }
      
      function exit( node ) {
        if ( !exited && other.isAfter ) {
          var depth = indexStack.length - 1;
          if ( depth === match && match + 1 === other.trail.indices.length ) {
            if ( other.trail.indices[match] === indexStack[depth] ) {
              match++;
              
              // once we have matched
              exited = true;
              if ( !excludeEndpoints ) {
                callbacks.exit( node );
              }
              return;
            }
          }
        }
        
        if ( entered && !exited ) {
          callbacks.exit( node );
        }
        
        if ( !entered && enterAfterNext ) {
          entered = true;
        }
        if ( !exited && exitAfterNext ) {
          exited = true;
        }
      }
      
      function push( index ) {
        var node = topNode().children[index];
        indexStack.push( index );
        nodeStack.push( node );
        enter( node );
      }
      
      function pop() {
        var index = indexStack.pop();
        var node = nodeStack.pop();
        exit( node );
        return index;
      }
      
      function topNode() {
        return nodeStack[nodeStack.length - 1];
      }
      
      function next( childrenOk ) {
        if ( childrenOk && topNode().children.length > 0 ) {
          // if there are children, walk to the first one
          push( 0 );
        } else {
          while ( indexStack.length > 0 ) {
            var index = pop();
            if ( topNode().children.length > index + 1 ) {
              push( index + 1 );
              break;
            }
          }
        }
      }
      
      // walk the starting trail
      for ( var i = 1; i < this.trail.length; i++ ) {
        push( this.trail.indices( i - 1 ) );
      }
      
      if ( this.isBefore ) {
        entered = true;
        if ( !excludeEndpoints ) {
          callbacks.enter( topNode() );
        }
      } else {
        if ( excludeEndpoints ) {
          enterAfterNext = true;
        } else {
          entered = true;
        }
      }
      next( this.isBefore );
      while ( !exited ) {
        next( true );
      }
    }
  };
  
})();

