// Copyright 2002-2012, University of Colorado

/**
 * Represents a trail (path in the graph) from a "root" node down to a descendant node.
 * In a DAG, or with different views, there can be more than one trail up from a node,
 * even to the same root node!
 *
 * This trail also mimics an Array, so trail[0] will be the root, and trail[trail.length-1]
 * will be the end node of the trail.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.Trail = function( nodes ) {
    this.nodes = [];
    this.length = 0;
    
    // indices[x] stores the index of nodes[x] in nodes[x-1]'s children
    this.indices = null;
    
    var trail = this;
    if ( nodes ) {
      if ( nodes instanceof scenery.Node ) {
        var node = nodes;
        
        // add just a single node in
        trail.addDescendant( node );
      } else {
        // process it as an array
        _.each( nodes, function( node ) {
          trail.addDescendant( node );
        } );
      }
    }
  };
  var Trail = scenery.Trail;
  
  Trail.prototype = {
    constructor: Trail,
    
    copy: function() {
      return new scenery.Trail( this.nodes.slice( 0 ) );
    },
    
    isEmpty: function() {
      return this.nodes.length === 0;
    },
    
    addAncestor: function( node ) {
      this.nodes.unshift( node );
      
      // mimic an Array
      this.length++;
      for ( var i = 0; i < this.length; i++ ) {
        this[i] = this.nodes[i];
      }
    },
    
    removeAncestor: function() {
      this.nodes.shift();
      
      // mimic an Array
      this.length--;
      delete this[this.length];
      for ( var i = 0; i < this.length; i++ ) {
        this[i] = this.nodes[i];
      }
    },
    
    addDescendant: function( node ) {
      this.nodes.push( node );
      
      // mimic an Array
      this.length++;
      this[this.length-1] = node;
    },
    
    removeDescendant: function() {
      this.nodes.pop();
      
      // mimic an Array
      this.length--;
      delete this[this.length];
    },
    
    // updates the indices array so that operations can depend on child indices
    reindex: function() {
      this.indices = [];
      for ( var i = 1; i < this.length; i++ ) {
        this.indices.push( _.indexOf( this.nodes[i-1].children, this.nodes[i] ) );
      }
    },
    
    equals: function( other ) {
      if ( this.length !== other.length ) {
        return false;
      }
      
      for ( var i = 0; i < this.nodes.length; i++ ) {
        if ( this.nodes[i] !== other.nodes[i] ) {
          return false;
        }
      }
      
      return true;
    },
    
    // whether this trail contains the complete 'other' trail, but with added descendants afterwards
    isExtensionOf: function( other ) {
      if ( this.length <= other.length ) {
        return false;
      }
      
      for ( var i = 0; i < other.nodes.length; i++ ) {
        if ( this.nodes[i] !== other.nodes[i] ) {
          return false;
        }
      }
      
      return true;
    },
    
    nodeFromTop: function( offset ) {
      return this.nodes[this.length - 1 - offset];
    },
    
    lastNode: function() {
      return this.nodeFromTop( 0 );
    },
    
    // returns the previous graph trail in the order of self-rendering
    previous: function() {
      if ( this.nodes.length <= 1 ) {
        return null;
      }
      
      var top = this.nodeFromTop( 0 );
      var parent = this.nodeFromTop( 1 );
      
      var parentIndex = _.indexOf( parent.children, top );
      var arr = this.nodes.slice( 0, this.nodes.length - 1 );
      if ( parentIndex === 0 ) {
        // we were the first child, so give it the trail to the parent
        return new Trail( arr );
      } else {
        // previous child
        arr.push( parent.children[parentIndex-1] );
        
        // and find its last terminal
        while( arr[arr.length-1].children.length !== 0 ) {
          var last = arr[arr.length-1];
          arr.push( last.children[last.children.length-1] );
        }
        
        return new Trail( arr );
      }
    },
    
    // in the order of self-rendering
    next: function() {
      var arr = this.nodes.slice( 0 );
      
      var top = this.nodeFromTop( 0 );
      if ( top.children.length > 0 ) {
        // if we have children, return the first child
        arr.push( top.children[0] );
        return new Trail( arr );
      } else {
        // walk down and attempt to find the next parent
        var depth = this.nodes.length - 1;
        
        while ( depth > 0 ) {
          var node = this.nodes[depth];
          var parent = this.nodes[depth-1];
          
          arr.pop(); // take off the node so we can add the next sibling if it exists
          
          var index = _.indexOf( parent.children, node );
          if ( index !== parent.children.length - 1 ) {
            // there is another (later) sibling. use that!
            arr.push( parent.children[index+1] );
            return new Trail( arr );
          } else {
            depth--;
          }
        }
        
        // if we didn't reach a later sibling by now, it doesn't exist
        return null;
      }
    },
    
    /*
     * Iterates between this trail and the other one, calling listener.enter( node ) and listener.exit( node ).
     * This will be bounded by listener.exit( startNode ) and listener.enter( endNode ),
     * unless the inclusive flag is true, then it will be bounded by listener.enter( startNode ) and listener.exit( endNode ).
     */
    eachBetween: function( other, listener, inclusive, reverseIfNecessary ) {
      var minPath = this;
      var maxPath = other;
      
      var exclusive = !inclusive;
      
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
      
      // bail out since one is a subtrail of the other (there is no 'between' from the above definition)
      if ( exclusive && splitIndex === minSharedLength ) {
        return;
      }
      
      // console.log( 'splitIndex: ' + splitIndex );
      
      // TODO: remove after debugging
      function dumpstr( node ) {
        var result = '|';
        while ( node.parents.length !== 0 ) {
          result = _.indexOf( node.parents[0].children, node ) + ',' + result;
          node = node.parents[0];
        }
        return result;
      }
      
      function recurse( node, depth, hasLowBound, hasHighBound ) {
        console.log( 'recurse: ' + dumpstr( node ) + ' ' + hasLowBound + ', ' + hasHighBound + ', ' + depth );
        if ( !hasLowBound && !hasHighBound ) {
          listener.enter( node );
          _.each( node.children, function( child ) {
            recurse( child, depth + 1, false, false );
          } );
          listener.exit( node );
        } else {
          // we are now assured that minPath.nodes[depth] !== maxPath.nodes[depth] (at least as subtrails), so each child is either high-bounded or low-bounded
          
          if ( !hasLowBound ) {
            listener.enter( node );
          }
          
          var lowIndex = hasLowBound ? _.indexOf( node.children, minPath.nodes[depth] ) : 0;
          var highIndex = hasHighBound ? _.indexOf( node.children, maxPath.nodes[depth] ) : node.children.length - 1;
          phet.assert( lowIndex !== -1, 'no low index' );
          phet.assert( highIndex !== -1, 'no high index' );
          
          console.log( 'lowIndex: ' + lowIndex + ', highIndex: ' + highIndex + ', depth: ' + depth + ', minPath.nodes.length: ' + minPath.nodes.length );
          for ( var i = lowIndex; i <= highIndex; i++ ) {
            var child = node.children[i];
            
            var isOnLowPath = hasLowBound && i === lowIndex;
            var isOnHighPath = hasHighBound && i === highIndex;
            
            // don't follow the subtree of a start node
            if ( isOnLowPath && minPath.nodes.length - 1 === depth ) {
              listener.exit( child );
              continue;
            }
            
            // don't follow the subtree of an end node
            if ( isOnHighPath && maxPath.nodes.length - 1 === depth ) {
              listener.enter( child );
              continue;
            }
            
            recurse( child, depth + 1, hasLowBound && i === lowIndex, hasHighBound && i === highIndex );
          }
          
          if ( !hasHighBound ) {
            listener.exit( node );
          }
        }
      }
      
      recurse( minPath.nodes[splitIndex-1], splitIndex, minPath.nodes.length !== splitIndex, maxPath.nodes.length !== splitIndex );
    },
    
    /* Standard Java-style compare. -1 means this trail is before (under) the other trail, 0 means equal, and 1 means this trail is
     * after (on top of) the other trail.
     * A shorter subtrail will compare as -1.
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
      
      // we scanned through and no nodes were different (one is a subtrail of the other)
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


