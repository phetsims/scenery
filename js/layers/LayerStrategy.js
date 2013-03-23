// Copyright 2002-2012, University of Colorado

/**
 * Controls the underlying layer behavior around a node. The node's LayerStrategy's enter() and exit() will be
 * called in a depth-first order during the layer building process, and will modify a LayerBuilder to signal any
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
    // true iff enter/exit will push/pop a layer type to the preferred stack. currently limited to only one layer type per level.
    hasPreferredLayerType: function( pointer, layerBuilder ) {
      return pointer.trail.lastNode().hasRenderer();
    },
    
    getPreferredLayerType: function( pointer, layerBuilder ) {
      assert && assert( this.hasPreferredLayerType( pointer, layerBuilder ) ); // sanity check
      
      var node = pointer.trail.lastNode();
      var preferredLayerType;
      
      if ( node.hasRendererLayerType() ) {
        preferredLayerType = node.getRendererLayerType();
      } else {
        preferredLayerType = layerBuilder.bestPreferredLayerTypeFor( [ node.getRenderer() ] );
        if ( !preferredLayerType ) {
          // there was no preferred layer type matching, just use the default
          preferredLayerType = node.getRenderer().defaultLayerType;
        }
      }
      
      return preferredLayerType;
    },
    
    enter: function( pointer, layerBuilder ) {
      var trail = pointer.trail;
      var node = trail.lastNode();
      var preferredLayerType;
      
      // if the node has a renderer, always push a layer type, so that we can pop on the exit() and ensure consistent behavior
      if ( node.hasRenderer() ) {
        preferredLayerType = this.getPreferredLayerType( pointer, layerBuilder );
        
        // push the preferred layer type
        layerBuilder.pushPreferredLayerType( preferredLayerType );
        if ( layerBuilder.getCurrentLayerType() !== preferredLayerType ) {
          layerBuilder.switchToType( pointer, preferredLayerType );
        }
      } else if ( node.hasSelf() ) {
        // node doesn't specify a renderer, but hasSelf.
        
        var supportedRenderers = node._supportedRenderers;
        var currentType = layerBuilder.getCurrentLayerType();
        preferredLayerType = layerBuilder.bestPreferredLayerTypeFor( supportedRenderers );
        
        // If any of the preferred types are compatible, use the top one. This allows us to support caching and hierarchical layer types
        if ( preferredLayerType ) {
          if ( currentType !== preferredLayerType ) {
            layerBuilder.switchToType( pointer, preferredLayerType );
          }
        } else {
          // if no preferred types are compatible, only switch if the current type is also incompatible
          if ( !currentType || !currentType.supportsNode( node ) ) {
            layerBuilder.switchToType( pointer, supportedRenderers[0].defaultLayerType );
          }
        }
      }
      
      if ( node.isLayerSplitBefore() || this.hasSplitFlags( node ) ) {
        layerBuilder.switchToType( pointer, layerBuilder.getCurrentLayerType() );
      }
      
      if ( node.hasSelf() ) {
        // trigger actual layer creation if necessary (allow collapsing of layers otherwise)
        layerBuilder.markSelf( pointer );
      }
    },
    
    // afterSelf: function( trail, layerBuilder ) {
    //   // no-op, and possibly not used
    // },
    
    // betweenChildren: function( trail, layerBuilder ) {
    //   // no-op, and possibly not used
    // },
    
    exit: function( pointer, layerBuilder ) {
      var trail = pointer.trail;
      var node = trail.lastNode();
      
      if ( node.hasRenderer() ) {
        layerBuilder.popPreferredLayerType();
        
        // switch down to the next lowest preferred layer type, if any. if null, pass the null to switchToType
        // this allows us to not 'leak' the renderer information, and the temporary layer type is most likely collapsed and ignored
        // NOTE: disabled for now, since this prevents us from having adjacent children sharing the same layer type
        // if ( layerBuilder.getCurrentLayerType() !== layerBuilder.getPreferredLayerType() ) {
        //   layerBuilder.switchToType( pointer, layerBuilder.getPreferredLayerType() );
        // }
      }
      
      if ( node.isLayerSplitAfter() || this.hasSplitFlags( node ) ) {
        layerBuilder.switchToType( pointer, layerBuilder.getCurrentLayerType() );
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
