// Copyright 2002-2012, University of Colorado

/**
 * A conceptual boundary between layers, where it is optional to have information about a previous or next layer.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.LayerBoundary = function() {
    // layer types before and after the boundary. null indicates the lack of information (first or last layer)
    this.previousLayerType = null;
    this.nextLayerType = null;
    
    // trails to the closest nodes with hasSelf() === true before and after the boundary
    this.previousSelfTrail = null;
    this.nextSelfTrail = null;
    
    // the TrailPointers where the previous layer was ended and the next layer begins (the trail, and enter() or exit())
    this.previousEndPointer = null;
    this.nextStartPointer = null;
  };
  var LayerBoundary = scenery.LayerBoundary;
  
  LayerBoundary.prototype = {
    constructor: LayerBoundary,
    
    hasPrevious: function() {
      return !!this.previousSelfTrail;
    },
    
    hasNext: function() {
      return !!this.nextSelfTrail;
    }
  };
  
  return LayerBoundary;
} );
