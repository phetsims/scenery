// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.GraphPath = function( nodes ) {
    // TODO: consider adding index information
    this.nodes = nodes || [];
  };
  var GraphPath = scenery.GraphPath;
  
  GraphPath.prototype = {
    constructor: GraphPath,
    
    copy: function() {
      return new scenery.GraphPath( this.nodes.slice( 0 ) );
    },
    
    isEmpty: function() {
      return this.nodes.length === 0;
    },
    
    addAncestor: function( node ) {
      this.nodes.unshift( node );
    },
    
    removeAncestor: function() {
      this.nodes.shift();
    },
    
    addDescendant: function( node ) {
      this.nodes.push( node );
    },
    
    removeDescendant: function() {
      this.nodes.pop();
    },
    
    equals: function( other ) {
      if ( this.nodes.length !== other.nodes.length ) {
        return false;
      }
      
      for ( var i = 0; i < this.nodes.length; i++ ) {
        if ( this.nodes[i] !== other.nodes[i] ) {
          return false;
        }
      }
      
      return true;
    },
    
    /*
     * Iterates between this graph path and the other one, calling listener.enter( node ) and listener.exit( node ).
     * This will be bounded by listener.exit( startNode ) and listener.enter( endNode )
     */
    eachBetween: function( other, listener, reverseIfNecessary ) {
      var minPath = this;
      var maxPath = other;
      
      if ( reverseIfNecessary ) {
        var comparison = this.compare( other );
        minPath = comparison === -1 ? this : other;
        maxPath = comparison === -1 ? other : this;
      }
      
      var minSharedLength = Math.min( minPath.nodes.length, maxPath.nodes.length );
      
      // set to the length so if we don't find a first split, it is after the shared region
      var splitIndex = minSharedLength;
      for ( var i = 0; i < minSharedLength; i++ ) {
        if ( minPath.nodes[i] !== maxPath.nodes[i] ) {
          splitIndex = i;
          break;
        }
      }
      
      function recurse( node, depth, hasLowBound, hasHighBound ) {
        if ( !hasLowBound && !hasHighBound ) {
          listener.enter( node );
          _.each( node.children, function( child ) {
            recurse( child, depth + 1, false, false );
          } );
          listener.exit( node );
        } else {
          // we are now assured that minPath.nodes[depth] !== maxPath.nodes[depth] (at least as subpaths), so each child is either high-bounded or low-bounded
          
          if ( !hasLowBound ) {
            listener.enter( node );
          }
          
          var passedLowBound = !hasLowBound;
          
          for ( var i = 0; i < node.children.length; i++ ) {
            var child = node.children[i];
            
            
            
            
            
            if ( !passedLowBound ) {
              // if we haven't passed the low bound, it MUST exist, and we are either (a) not rendered, or (b) low-bounded
              
              if ( startPath[depth] === child ) {
                // only recursively render if the switch is "before" the node
                if ( startPath[depth]._layerBeforeRender === layer ) {
                  recursivePartialRender( child, depth + 1, startPath.length !== depth, false ); // if it has a low-bound, it can't have a high bound
                }
              
                // for later children, we have passed the low bound.
                passedLowBound = true;
              }
              
              // either way, go on to the next nodes
              continue;
            }
            
            if ( hasHighBound && endPath[depth] === child ) {
              // only recursively render if the switch is "after" the node
              if ( endPath[depth]._layerAfterRender === layer ) {
                // high-bounded here
                recursivePartialRender( child, depth + 1, false, endPath.length !== depth ); // if it has a high-bound, it can't have a low bound
              }
              
              // don't render any more children, since we passed the high bound
              break;
            }
            
            
            
            
            
          }
          if ( !hasHighBound ) {
            listener.exit( node );
          }
        }
      }
      
      recurse( minPath[splitIndex-1], splitIndex, minPath.nodes.length !== splitIndex, maxPath.nodes.length !== splitIndex );
    },
    
    /* Standard Java-style compare. -1 means this graph path is before (under) the other path, 0 means equal, and 1 means this path is
     * after (on top of) the other path.
     * A shorter subpath will compare as -1.
     */
    compare: function( other ) {
      phet.assert( !this.isEmpty() );
      phet.assert( !other.isEmpty() );
      phet.assert( this.nodes[0] === other.nodes[0] );
      
      var minIndex = Math.min( this.nodes.length, other.nodes.length );
      for ( var i = 1; i < minIndex; i++ ) {
        if ( this.nodes[i] !== other.nodes[i] ) {
          var myIndex = _.indexOf( this.nodes[i-1].children, this.nodes[i] );
          var otherIndex = _.indexOf( other.nodes[i-1].children, other.nodes[i] );
          phet.assert( myIndex !== otherIndex ); // they should be different if the nodes are different
          if ( myIndex < otherIndex ) {
            return -1;
          } else {
            return 1;
          }
        }
      }
      
      // we scanned through and no nodes were different (one is a subpath of the other)
      if ( this.nodes.length < other.nodes.length ) {
        return -1;
      } else if ( this.nodes.length > other.nodes.length ) {
        return 1;
      } else {
        return 0;
      }
    }
  };
  
})();


