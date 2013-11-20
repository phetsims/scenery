// Copyright 2002-2013, University of Colorado

/**
 * API for RenderState
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.RenderState = function RenderState() {
    
  };
  var RenderState = scenery.RenderState;
  
  inherit( Object, RenderState, {
    isBackbone: function() {
      
    },
    
    isCanvasCache: function() {
      
    },
    
    isCacheShared: function() {
      // true/false
    },
    
    requestsSplit: function() {
      
    },
    
    getStateForDescendant: function( trail ) {
      // new state
    },
    
    getPaintedRenderer: function() {
      
    },
    
    // renderer for the (Canvas) cache
    getCacheRenderer: function() {
      
    }
  } );
  
  return RenderState;
} );
