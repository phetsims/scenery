// Copyright 2002-2012, University of Colorado

/**
 * Mutable state passed through the scene graph rendering process that stores
 * the current transformation and layer.
 *
 * A fresh RenderState should be used for each render pass.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  scenery.RenderState = function( scene ) {
    this.layer = null;
    this.scene = scene;
    
    // clipping shapes should be added in reference to the global coordinate frame
    this.clipShapes = [];
    
    // when non-null, children not intersecting the global bounds here may not be rendered for efficiency
    this.childRestrictedBounds = null;
  };

  var RenderState = scenery.RenderState;

  RenderState.prototype = {
    constructor: RenderState,
    
    // add a clipping region in the global coordinate frame (relative to the root of a trail)
    pushClipShape: function( shape ) {
      // transform the shape into global coordinates
      this.clipShapes.push( shape );
      
      // notify the layer to actually do the clipping
      if ( this.layer ) {
        this.layer.pushClipShape( shape );
      }
    },
    
    popClipShape: function() {
      this.clipShapes.pop();
      
      if ( this.layer ) {
        this.layer.popClipShape();
      }
    },
    
    isCanvasState: function() {
      return this.layer.isCanvasLayer;
    },
    
    isDOMState: function() {
      return this.layer.isDOMLayer;
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      this.layer.applyTransformationMatrix( matrix );
    }
  };
} );
