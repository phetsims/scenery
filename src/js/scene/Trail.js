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
    if ( nodes instanceof scenery.Trail ) {
      // copy constructor (takes advantage of already built index information)
      var otherTrail = nodes;
      
      this.nodes = otherTrail.nodes.slice( 0 );
      this.length = otherTrail.length;
      this.indices = otherTrail.indices.slice( 0 );
      return;
    }
    
    this.nodes = [];
    this.length = 0;
    
    // indices[x] stores the index of nodes[x] in nodes[x-1]'s children
    this.indices = [];
    
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
      return new Trail( this );
    },
    
    get: function( index ) {
      if ( index >= 0 ) {
        return this.nodes[index];
      } else {
        // negative index goes from the end of the array
        return this.nodes[this.nodes.length + index];
      }
    },
    
    isEmpty: function() {
      return this.nodes.length === 0;
    },
    
    addAncestor: function( node, index ) {
      var oldRoot = this.nodes[0];
      
      this.nodes.unshift( node );
      if ( oldRoot ) {
        this.indices.unshift( index === undefined ? _.indexOf( node.children, oldRoot ) : index );
      }
      
      // mimic an Array
      this.length++;
      for ( var i = 0; i < this.length; i++ ) {
        this[i] = this.nodes[i];
      }
    },
    
    removeAncestor: function() {
      this.nodes.shift();
      if ( this.indices.length ) {
        this.indices.shift();
      }
      
      // mimic an Array
      this.length--;
      delete this[this.length];
      for ( var i = 0; i < this.length; i++ ) {
        this[i] = this.nodes[i];
      }
    },
    
    addDescendant: function( node, index ) {
      var parent = this.lastNode();
      
      this.nodes.push( node );
      if ( parent ) {
        this.indices.push( index === undefined ? _.indexOf( parent.children, node ) : index );
      }
      
      // mimic an Array
      this.length++;
      this[this.length-1] = node;
    },
    
    removeDescendant: function() {
      this.nodes.pop();
      if ( this.indices.length ) {
        this.indices.pop();
      }
      
      // mimic an Array
      this.length--;
      delete this[this.length];
    },
    
    // refreshes the internal index references (important if any children arrays were modified!)
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
    
    // TODO: phase out in favor of get()
    nodeFromTop: function( offset ) {
      return this.nodes[this.length - 1 - offset];
    },
    
    lastNode: function() {
      return this.nodeFromTop( 0 );
    },
    
    rootNode: function() {
      return this.nodes[0];
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
    
    /* Standard Java-style compare. -1 means this trail is before (under) the other trail, 0 means equal, and 1 means this trail is
     * after (on top of) the other trail.
     * A shorter subtrail will compare as -1.
     */
    compare: function( other ) {
      phet.assert( !this.isEmpty(), 'cannot compare with an empty trail' );
      phet.assert( !other.isEmpty(), 'cannot compare with an empty trail' );
      phet.assert( this.nodes[0] === other.nodes[0], 'for Trail comparison, trails must have the same root node' );
      
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
    },
    
    toString: function() {
      this.reindex();
      if ( !this.length ) {
        return 'Empty Trail';
      }
      return '[Trail ' + this.indices.join( '.' ) + ']';
    }
  };
  
})();


