// Copyright 2002-2012, University of Colorado

/**
 * Controls the underlying layer behavior around a node. The node's LayerStrategy's enter() and exit() will be
 * called in a depth-first order during the layer building process, and will modify a LayerState to signal any
 * layer-specific signals.
 *
 * This generally ensures that a layer containing the proper renderer and settings to support its associated node
 * will be created.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  /*
   * If the node specifies a renderer, we will always push a preferred type. That type will be fresh (if rendererOptions are specified), otherwise
   * the top matching preferred type for that renderer will be used. This allows us to always pop in the exit().
   *
   * Specified as such, since there is no needed shared state (we can have node.layerStrategy = scenery.LayerStrategy for many nodes)
   */
  scenery.LayerStrategy = {
    enter: function( trail, layerState ) {
      var node = trail.lastNode();
      var preferredLayerType;
      
      // if the node has a renderer, always push a layer type, so that we can pop on the exit() and ensure consistent behavior
      if ( node.hasRenderer() ) {
        if ( node.hasRendererLayerType() ) {
          preferredLayerType = node.getRendererLayerType();
        } else {
          preferredLayerType = layerState.bestPreferredLayerTypeFor( [ node.getRenderer() ] );
          if ( !preferredLayerType ) {
            // there was no preferred layer type matching, just use the default
            preferredLayerType = node.getRenderer().defaultLayerType;
          }
        }
        
        // push the preferred layer type
        layerState.pushPreferredLayerType( preferredLayerType );
        if ( layerState.getCurrentLayerType() !== preferredLayerType ) {
          layerState.switchToType( trail, preferredLayerType );
        }
      } else if ( node.hasSelf() ) {
        // node doesn't specify a renderer, but hasSelf.
        
        var supportedRenderers = node._supportedRenderers;
        var currentType = layerState.getCurrentLayerType();
        preferredLayerType = layerState.bestPreferredLayerTypeFor( supportedRenderers );
        
        // If any of the preferred types are compatible, use the top one. This allows us to support caching and hierarchical layer types
        if ( preferredLayerType ) {
          if ( currentType !== preferredLayerType ) {
            layerState.switchToType( trail, preferredLayerType );
          }
        } else {
          // if no preferred types are compatible, only switch if the current type is also incompatible
          if ( !currentType || !currentType.supportsNode( node ) ) {
            layerState.switchToType( trail, supportedRenderers[0].defaultLayerType );
          }
        }
      }
      
      if ( node.isLayerSplitBefore() || this.hasSplitFlags( node ) ) {
        layerState.switchToType( trail, layerState.getCurrentLayerType() );
      }
      
      if ( node.hasSelf() ) {
        // trigger actual layer creation if necessary (allow collapsing of layers otherwise)
        layerState.markSelf();
      }
    },
    
    // afterSelf: function( trail, layerState ) {
    //   // no-op, and possibly not used
    // },
    
    // betweenChildren: function( trail, layerState ) {
    //   // no-op, and possibly not used
    // },
    
    exit: function( trail, layerState ) {
      var node = trail.lastNode();
      
      if ( node.hasRenderer() ) {
        layerState.popPreferredLayerType();
        
        // switch down to the next lowest preferred layer type, if any. if null, pass the null to switchToType
        // this allows us to not 'leak' the renderer information, and the temporary layer type is most likely collapsed and ignored
        // NOTE: disabled for now, since this prevents us from having adjacent children sharing the same layer type
        // if ( layerState.getCurrentLayerType() !== layerState.getPreferredLayerType() ) {
        //   layerState.switchToType( trail, layerState.getPreferredLayerType() );
        // }
      }
      
      if ( node.isLayerSplitAfter() || this.hasSplitFlags( node ) ) {
        layerState.switchToType( trail, layerState.getCurrentLayerType() );
      }
    },
    
    // whether splitting before and after the node is required
    hasSplitFlags: function( node ) {
      // currently, only enforce splitting if we are using CSS transforms
      var rendererOptions = node.getRendererOptions();
      return node.hasRenderer() && rendererOptions && (
        rendererOptions.cssTranslation ||
        rendererOptions.cssRotation ||
        rendererOptions.cssScale ||
        rendererOptions.cssTransform
      );
    }
  };
  var LayerStrategy = scenery.LayerStrategy;
  
  return LayerStrategy;
} );
