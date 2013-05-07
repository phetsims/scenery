// Copyright 2002-2012, University of Colorado

/**
 * An Instance of a Node in the expanded tree form.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Instance = function( trail, scene, layer ) {
    this.trail = trail;
    this.scene = scene;
    this.layer = layer;
    
    // TODO: ensure that we can track this? otherwise remove it for memory and speed
    this.parent = null;
    this.children = [];
    
    // TODO: track these? should significantly accelerate subtree-changing operations
    this.startAffectedLayer = null;
    this.endAffectedLayer = null;
  };
  var Instance = scenery.Instance;
  
  Instance.prototype = {
    constructor: Instance,
    
    reindex: function() {
      this.trail.reindex();
    },
    
    equals: function( other ) {
      assert && assert( ( this === other ) === this.trail.equals( other.trail ), 'We assume a 1-1 mapping from trails to instances' );
      return this === other;
    }
  };
  
  return Instance;
} );


