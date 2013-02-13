// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.GraphPath = function() {
    // TODO: consider adding index information
    this.nodes = [];
  };
  var GraphPath = scenery.GraphPath;
  
  GraphPath.prototype = {
    constructor: GraphPath,
    
    isEmpty: function() {
      return this.nodes.length === 0;
    },
    
    addAncestor: function( node ) {
      this.nodes.unshift( node );
    },
    
    addDescendant: function( node ) {
      this.nodes.push( node );
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


