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

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  var assertExtra = require( 'ASSERT/assert' )( 'scenery.extra', false );
  
  var Transform3 = require( 'DOT/Transform3' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  require( 'SCENERY/nodes/Node' );
  // require( 'SCENERY/util/TrailPointer' );
  
  scenery.Trail = function( nodes ) {
    /*
     * Controls the immutability of the trail.
     * If set to true, add/remove descendant/ancestor should fail if assertions are enabled
     * Use setImmutable() or setMutable() to signal a specific type of protection, so it cannot be changed later
     */
    this.immutable = undefined;
    
    if ( nodes instanceof Trail ) {
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
    
    // convenience function to determine whether this trail will render something
    isPainted: function() {
      return this.lastNode().isPainted();
    },
    
    get: function( index ) {
      if ( index >= 0 ) {
        return this.nodes[index];
      } else {
        // negative index goes from the end of the array
        return this.nodes[this.nodes.length + index];
      }
    },
    
    slice: function( startIndex, endIndex ) {
      return new Trail( this.nodes.slice( startIndex, endIndex ) );
    },
    
    subtrailTo: function( node, excludeNode ) {
      return this.slice( 0, _.indexOf( this.nodes, node ) + ( excludeNode ? 0 : 1 ) );
    },
    
    isEmpty: function() {
      return this.nodes.length === 0;
    },
    
    getTransform: function() {
      // always return a defensive copy of a transform
      var transform = new Transform3();
      
      // from the root up
      _.each( this.nodes, function( node ) {
        transform.appendTransform( node._transform );
      } );
      
      return transform;
    },
    
    addAncestor: function( node, index ) {
      assert && assert( !this.immutable, 'cannot modify an immutable Trail with addAncestor' );
      
      var oldRoot = this.nodes[0];
      
      this.nodes.unshift( node );
      if ( oldRoot ) {
        this.indices.unshift( index === undefined ? _.indexOf( node._children, oldRoot ) : index );
      }
      
      // mimic an Array
      this.length++;
      return this;
    },
    
    removeAncestor: function() {
      assert && assert( !this.immutable, 'cannot modify an immutable Trail with removeAncestor' );
      
      this.nodes.shift();
      if ( this.indices.length ) {
        this.indices.shift();
      }
      
      // mimic an Array
      this.length--;
      return this;
    },
    
    addDescendant: function( node, index ) {
      assert && assert( !this.immutable, 'cannot modify an immutable Trail with addDescendant' );
      
      var parent = this.lastNode();
      
      this.nodes.push( node );
      if ( parent ) {
        this.indices.push( index === undefined ? _.indexOf( parent._children, node ) : index );
      }
      
      // mimic an Array
      this.length++;
      return this;
    },
    
    removeDescendant: function() {
      assert && assert( !this.immutable, 'cannot modify an immutable Trail with removeDescendant' );
      
      this.nodes.pop();
      if ( this.indices.length ) {
        this.indices.pop();
      }
      
      // mimic an Array
      this.length--;
      return this;
    },
    
    // refreshes the internal index references (important if any children arrays were modified!)
    reindex: function() {
      for ( var i = 1; i < this.length; i++ ) {
        // only replace indices where they have changed (this was a performance hotspot)
        var currentIndex = this.indices[i-1];
        if ( this.nodes[i-1]._children[currentIndex] !== this.nodes[i] ) {
          this.indices[i-1] = _.indexOf( this.nodes[i-1]._children, this.nodes[i] );
        }
      }
    },
    
    setImmutable: function() {
      assert && assert( this.immutable !== false, 'A trail cannot be made immutable after being flagged as mutable' );
      
      this.immutable = true;
    },
    
    setMutable: function() {
      assert && assert( this.immutable !== true, 'A trail cannot be made mutable after being flagged as immutable' );
      
      this.immutable = false;
    },
    
    areIndicesValid: function() {
      for ( var i = 1; i < this.length; i++ ) {
        var currentIndex = this.indices[i-1];
        if ( this.nodes[i-1]._children[currentIndex] !== this.nodes[i] ) {
          return false;
        }
      }
      return true;
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
    isExtensionOf: function( other, allowSameTrail ) {
      assertExtra && assertExtra( this.areIndicesValid(), 'Trail.compare this.areIndicesValid() failed' );
      assertExtra && assertExtra( other.areIndicesValid(), 'Trail.compare other.areIndicesValid() failed' );
      
      if ( this.length <= other.length - ( allowSameTrail ? 1 : 0 ) ) {
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
      
      var parentIndex = _.indexOf( parent._children, top );
      assert && assert( parentIndex !== -1 );
      var arr = this.nodes.slice( 0, this.nodes.length - 1 );
      if ( parentIndex === 0 ) {
        // we were the first child, so give it the trail to the parent
        return new Trail( arr );
      } else {
        // previous child
        arr.push( parent._children[parentIndex-1] );
        
        // and find its last terminal
        while( arr[arr.length-1]._children.length !== 0 ) {
          var last = arr[arr.length-1];
          arr.push( last._children[last._children.length-1] );
        }
        
        return new Trail( arr );
      }
    },
    
    // like previous(), but keeps moving back until the trail goes to a node with isPainted() === true
    previousPainted: function() {
      var result = this.previous();
      while ( result && !result.isPainted() ) {
        result = result.previous();
      }
      return result;
    },
    
    // in the order of self-rendering
    next: function() {
      var arr = this.nodes.slice( 0 );
      
      var top = this.nodeFromTop( 0 );
      if ( top._children.length > 0 ) {
        // if we have children, return the first child
        arr.push( top._children[0] );
        return new Trail( arr );
      } else {
        // walk down and attempt to find the next parent
        var depth = this.nodes.length - 1;
        
        while ( depth > 0 ) {
          var node = this.nodes[depth];
          var parent = this.nodes[depth-1];
          
          arr.pop(); // take off the node so we can add the next sibling if it exists
          
          var index = _.indexOf( parent._children, node );
          if ( index !== parent._children.length - 1 ) {
            // there is another (later) sibling. use that!
            arr.push( parent._children[index+1] );
            return new Trail( arr );
          } else {
            depth--;
          }
        }
        
        // if we didn't reach a later sibling by now, it doesn't exist
        return null;
      }
    },
    
    // like next(), but keeps moving back until the trail goes to a node with isPainted() === true
    nextPainted: function() {
      var result = this.next();
      while ( result && !result.isPainted() ) {
        result = result.next();
      }
      return result;
    },
    
    // calls callback( trail ) for this trail, and each descendant trail
    eachTrailUnder: function( callback ) {
      new scenery.TrailPointer( this, true ).eachTrailBetween( new scenery.TrailPointer( this, false ), callback );
    },
    
    /*
     * Standard Java-style compare. -1 means this trail is before (under) the other trail, 0 means equal, and 1 means this trail is
     * after (on top of) the other trail.
     * A shorter subtrail will compare as -1.
     *
     * Assumes that the Trails are properly indexed. If not, please reindex them!
     *
     * Comparison is for the rendering order, so an ancestor is 'before' a descendant
     */
    compare: function( other ) {
      assert && assert( !this.isEmpty(), 'cannot compare with an empty trail' );
      assert && assert( !other.isEmpty(), 'cannot compare with an empty trail' );
      assert && assert( this.nodes[0] === other.nodes[0], 'for Trail comparison, trails must have the same root node' );
      assertExtra && assertExtra( this.areIndicesValid(), 'Trail.compare this.areIndicesValid() failed on ' + this.toString() );
      assertExtra && assertExtra( other.areIndicesValid(), 'Trail.compare other.areIndicesValid() failed on ' + other.toString() );
      
      var minNodeIndex = Math.min( this.indices.length, other.indices.length );
      for ( var i = 0; i < minNodeIndex; i++ ) {
        if ( this.indices[i] !== other.indices[i] ) {
          if ( this.indices[i] < other.indices[i] ) {
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
    
    localToGlobalPoint: function( point ) {
      return this.getTransform().transformPosition2( point );
    },
    
    localToGlobalBounds: function( bounds ) {
      return this.getTransform().transformBounds2( bounds );
    },
    
    globalToLocalPoint: function( point ) {
      return this.getTransform().inversePosition2( point );
    },
    
    globalToLocalBounds: function( bounds ) {
      return this.getTransform().inverseBounds2( bounds );
    },
    
    // concatenates the unique IDs of nodes in the trail, so that we can do id-based lookups
    getUniqueId: function() {
      // TODO: consider caching this if it is ever a bottleneck. it seems like it might be called in layer-refresh inner loops
      return _.map( this.nodes, function( node ) { return node.getId(); } ).join( '-' );
    },
    
    toString: function() {
      this.reindex();
      if ( !this.length ) {
        return 'Empty Trail';
      }
      return '[Trail ' + this.indices.join( '.' ) + ' ' + this.getUniqueId() + ']';
    }
  };
  
  // like eachTrailBetween, but only fires for painted trails
  Trail.eachPaintedTrailbetween = function( a, b, callback, excludeEndTrails, scene ) {
    Trail.eachTrailBetween( a, b, function( trail ) {
      if ( trail && trail.isPainted() ) {
        callback( trail );
      }
    }, excludeEndTrails, scene );
  };
  
  // global way of iterating across trails
  Trail.eachTrailBetween = function( a, b, callback, excludeEndTrails, scene ) {
    var aPointer = a ? new scenery.TrailPointer( a.copy(), true ) : new scenery.TrailPointer( new scenery.Trail( scene ), true );
    var bPointer = b ? new scenery.TrailPointer( b.copy(), true ) : new scenery.TrailPointer( new scenery.Trail( scene ), false );
    
    // if we are excluding endpoints, just bump the pointers towards each other by one step
    if ( excludeEndTrails ) {
      aPointer.nestedForwards();
      bPointer.nestedBackwards();
      
      // they were adjacent, so no callbacks will be executed
      if ( aPointer.compareNested( bPointer ) === 1 ) {
        return;
      }
    }
    
    aPointer.depthFirstUntil( bPointer, function( pointer ) {
      if ( pointer.isBefore ) {
        callback( pointer.trail );
      }
    }, false );
  };
  
  return Trail;
} );


