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
    
    // trails to the closest nodes with isPainted() === true before and after the boundary
    this.previousPaintedTrail = null;
    this.nextPaintedTrail = null;
    
    // the TrailPointers where the previous layer was ended and the next layer begins (the trail, and enter() or exit())
    this.previousEndPointer = null;
    this.nextStartPointer = null;
  };
  var LayerBoundary = scenery.LayerBoundary;
  
  LayerBoundary.prototype = {
    constructor: LayerBoundary,
    
    hasPrevious: function() {
      return !!this.previousPaintedTrail;
    },
    
    hasNext: function() {
      return !!this.nextPaintedTrail;
    },
    
    // assumes that trail is reindexed
    equivalentPreviousTrail: function( trail ) {
      if ( this.previousPaintedTrail && trail ) {
        this.previousPaintedTrail.reindex();
        return this.previousPaintedTrail.equals( trail );
      } else {
        // check that handles null versions properly
        return this.previousPaintedTrail === trail;
      }
    },
    
    equivalentNextTrail: function( trail ) {
      if ( this.nextPaintedTrail && trail ) {
        this.nextPaintedTrail.reindex();
        return this.nextPaintedTrail.equals( trail );
      } else {
        // check that handles null versions properly
        return this.nextPaintedTrail === trail;
      }
    },
    
    toString: function() {
      return 'boundary:' +
             '\n    types:    ' +
                  ( this.previousLayerType ? this.previousLayerType.name : '' ) +
                  ' => ' +
                  ( this.nextLayerType ? this.nextLayerType.name : '' ) +
             '\n    trails:   ' +
                  ( this.previousPaintedTrail ? this.previousPaintedTrail.getUniqueId() : '' ) +
                  ' => ' +
                  ( this.nextPaintedTrail ? this.nextPaintedTrail.getUniqueId() : '' ) +
             '\n    pointers: ' +
                  ( this.previousEndPointer ? this.previousEndPointer.toString() : '' ) +
                  ' => ' +
                  ( this.nextStartPointer ? this.nextStartPointer.toString() : '' );
    }
  };
  
  return LayerBoundary;
} );
