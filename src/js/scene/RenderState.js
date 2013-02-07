// Copyright 2002-2012, University of Colorado

/**
 * Mutable state passed through the scene graph rendering process that stores
 * the current transformation and layer.
 *
 * A fresh RenderState should be used for each render pass.
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
  "use strict";
  
  phet.scene.RenderState = function( scene ) {
    this.transform = new phet.math.Transform3();
    
    this.layer = null;
    this.scene = scene;
    
    // clipping shapes should be added in reference to the global coordinate frame
    this.clipShapes = [];
    
    // whether to allow switching layers mid-render
    this.multiLayerRender = true;
    
    // when non-null, children not intersecting the global bounds here may not be rendered for efficiency
    this.childRestrictedBounds = null;
  };

  var RenderState = phet.scene.RenderState;

  RenderState.prototype = {
    constructor: RenderState,
    
    // add a clipping region in the local coordinate frame
    pushClipShape: function( shape ) {
      // transform the shape into global coordinates
      this.clipShapes.push( this.transform.transformShape( shape ) );
      
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
    
    switchToLayer: function( layer ) {
      // don't change layers if it's not supported (gracefully handles single-layer rendering at a time)
      if ( this.layer && !this.multiLayerRender ) {
        return;
      }
      
      if ( this.layer ) {
        this.layer.cooldown();
      }
      
      this.layer = layer;
      
      // give the layer the current state so it can initialize itself properly
      layer.initialize( this );
    },
    
    finish: function() {
      if ( this.layer ) {
        this.layer.cooldown();
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
      this.transform.append( matrix );
      this.layer.applyTransformationMatrix( matrix );
    }
  };
})();
