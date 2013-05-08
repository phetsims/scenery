// Copyright 2002-2012, University of Colorado

/**
 * A Canvas-backed layer in the scene graph. Each layer handles dirty-region handling separately,
 * and corresponds to a single canvas / svg element / DOM element in the main container.
 * Importantly, it does not contain rendered content from a subtree of the main
 * scene graph. It only will render a contiguous block of nodes visited in a depth-first
 * manner.
 *
 * Backing store pixel ratio info: http://www.html5rocks.com/en/tutorials/canvas/hidpi/
 *
 * TODO: update internal documentation
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Shape = require( 'KITE/Shape' );
  
  var Layer = require( 'SCENERY/layers/Layer' ); // uses Layer's prototype for inheritance
  require( 'SCENERY/util/CanvasContextWrapper' );
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/TrailPointer' );
  require( 'SCENERY/util/Util' );
  
  // assumes main is wrapped with JQuery
  /*
   *
   */
  scenery.CanvasLayer = function( args ) {
    sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' constructor' );
    Layer.call( this, args );
    
    // TODO: deprecate Scene's backing scale, and handle this on a layer-by-layer option?
    this.backingScale = args.scene.backingScale;
    if ( args.fullResolution !== undefined ) {
      this.backingScale = args.fullResolution ? scenery.Util.backingScale( document.createElement( 'canvas' ).getContext( '2d' ) ) : 1;
    }
    
    this.logicalWidth = this.scene.sceneBounds.width;
    this.logicalHeight = this.scene.sceneBounds.height;
    
    var canvas = document.createElement( 'canvas' );
    canvas.width = this.logicalWidth * this.backingScale;
    canvas.height = this.logicalHeight * this.backingScale;
    canvas.style.width = this.logicalWidth + 'px';
    canvas.style.height = this.logicalHeight + 'px';
    canvas.style.position = 'absolute';
    canvas.style.left = '0';
    canvas.style.top = '0';
    
    // add this layer on top (importantly, the constructors of the layers are called in order)
    this.$main.append( canvas );
    
    this.canvas = canvas;
    // this.context = new DebugContext( canvas.getContext( '2d' ) );
    this.context = canvas.getContext( '2d' );
    this.scene = args.scene;
    
    // workaround for Chrome (WebKit) miterLimit bug: https://bugs.webkit.org/show_bug.cgi?id=108763
    this.context.miterLimit = 20;
    this.context.miterLimit = 10;
    
    this.isCanvasLayer = true;
    
    this.wrapper = new scenery.CanvasContextWrapper( this.canvas, this.context );
    
    this.boundlessCount = 0; // count of how many trails do not support bounds. we only will use dirty region repainting if this number is 0.
  };
  var CanvasLayer = scenery.CanvasLayer;
  
  inherit( CanvasLayer, Layer, {
    
    /*
     * Renders the canvas layer from the scene
     *
     * Supported args: {
     *   fullRender: true, // disables drawing to just dirty rectangles
     *   TODO: pruning with bounds and flag to disable
     * }
     */
    render: function( scene, args ) {
      args = args || {};
      
      var dirtyBoundsEnabled = this.canUseDirtyRegions() && !args.fullRender;
      
      // bail out quickly if possible
      if ( dirtyBoundsEnabled && this.dirtyBounds.isEmpty() ) {
        return;
      }
      
      // switch to an identity transform
      this.context.setTransform( this.backingScale, 0, 0, this.backingScale, 0, 0 );
      
      var visibleDirtyBounds = dirtyBoundsEnabled ? this.dirtyBounds.intersection( scene.sceneBounds ) : scene.sceneBounds;
      
      if ( !visibleDirtyBounds.isEmpty() ) {
        this.clearGlobalBounds( visibleDirtyBounds );
        
        if ( dirtyBoundsEnabled ) {
          this.pushClipShape( Shape.bounds( visibleDirtyBounds ) );
        }
        
        // dirty bounds (clear, possibly set restricted bounds and handling for that)
        // visibility checks
        this.recursiveRender( scene, args );
        
        // exists for now so that we pop the necessary context state
        if ( dirtyBoundsEnabled ) {
          this.popClipShape();
        }
      }
      
      // we rendered everything, no more dirty bounds
      this.dirtyBounds = Bounds2.NOTHING;
    },
    
    recursiveRender: function( scene, args ) {
      var layer = this;
      var i;
      var startPointer = new scenery.TrailPointer( this.startPaintedTrail, true );
      var endPointer = new scenery.TrailPointer( this.endPaintedTrail, true );
      
      // stack for canvases that need to be painted, since some effects require scratch canvases
      var wrapperStack = [ this.wrapper ]; // type {CanvasContextWrapper}
      this.wrapper.resetStyles(); // let's be defensive, save() and restore() may have been called previously
      
      function requiresScratchCanvas( trail ) {
        return trail.lastNode().getOpacity() < 1;
      }
      
      function getCanvasWrapper() {
        // TODO: verify that this works with hi-def canvases
        // TODO: use a cache of scratch canvases/contexts on the scene for this purpose, instead of creation
        var canvas = document.createElement( 'canvas' );
        canvas.width = layer.logicalWidth * layer.backingScale;
        canvas.height = layer.logicalHeight * layer.backingScale;
        // $( canvas ).css( 'width', layer.logicalWidth );
        // $( canvas ).css( 'height', layer.logicalHeight );
        var context = canvas.getContext( '2d' );
        
        return new scenery.CanvasContextWrapper( canvas, context );
      }
      
      function topWrapper() {
        return wrapperStack[wrapperStack.length-1];
      }
      
      function enter( trail ) {
        var node = trail.lastNode();
        
        if ( requiresScratchCanvas( trail ) ) {
          var wrapper = getCanvasWrapper();
          wrapperStack.push( wrapper );
          
          var newContext = wrapper.context;
          
          // switch to an identity transform
          newContext.setTransform( layer.backingScale, 0, 0, layer.backingScale, 0, 0 );
          
          // properly set the necessary transform on the context
          _.each( trail.nodes, function( node ) {
            node.transform.getMatrix().canvasAppendTransform( newContext );
          } );
        } else {
          node.transform.getMatrix().canvasAppendTransform( topWrapper().context );
        }
        
        if ( node._clipShape ) {
          // TODO: move to wrapper-specific part
          layer.pushClipShape( node._clipShape );
        }
      }
      
      function exit( trail ) {
        var node = trail.lastNode();
        
        if ( node._clipShape ) {
          // TODO: move to wrapper-specific part
          layer.popClipShape();
        }
        
        if ( requiresScratchCanvas( trail ) ) {
          var baseContext = wrapperStack[wrapperStack.length-2].context;
          var topCanvas = wrapperStack[wrapperStack.length-1].canvas;
          
          // apply necessary style transforms before painting our popped canvas onto the next canvas
          var opacityChange = trail.lastNode().getOpacity() < 1;
          if ( opacityChange ) {
            baseContext.globalAlpha = trail.lastNode().getOpacity();
          }
          
          // paint our canvas onto the level below with a straight transform
          baseContext.save();
          baseContext.setTransform( 1, 0, 0, 1, 0, 0 );
          baseContext.drawImage( topCanvas, 0, 0 );
          baseContext.restore();
          
          // reset styles
          if ( opacityChange ) {
            baseContext.globalAlpha = 1;
          }
          
          wrapperStack.pop();
        } else {
          node.transform.getInverse().canvasAppendTransform( topWrapper().context );
        }
      }
      
      /*
       * We count how many invisible nodes are in our trail, so we can properly iterate without inspecting everything.
       * Additionally, state changes (enter/exit) are only done when nodes are visible, so we skip overhead. If
       * invisibleCount > 0, then the current node is invisible.
       */
      var invisibleCount = 0;
      
      var boundaryTrail;
      
      // sanity check, and allows us to get faster speed
      startPointer.trail.reindex();
      endPointer.trail.reindex();
      
      // first, we need to walk the state up to before our pointer (as far as the recursive handling is concerned)
      // if the pointer is 'before' the node, don't call its enterState since this will be taken care of as the first step.
      // if the pointer is 'after' the node, call enterState since it will call exitState immediately inside the loop
      var startWalkLength = startPointer.trail.length - ( startPointer.isBefore ? 1 : 0 );
      boundaryTrail = new scenery.Trail();
      for ( i = 0; i < startWalkLength; i++ ) {
        var startNode = startPointer.trail.nodes[i];
        boundaryTrail.addDescendant( startNode );
        invisibleCount += startNode.isVisible() ? 0 : 1;
        
        if ( invisibleCount === 0 ) {
          // walk up initial state
          enter( boundaryTrail );
        }
      }
      
      startPointer.depthFirstUntil( endPointer, function renderPointer( pointer ) {
        // handle render here
        
        var node = pointer.trail.lastNode();
        
        if ( pointer.isBefore ) {
          invisibleCount += node.isVisible() ? 0 : 1;
          
          if ( invisibleCount === 0 ) {
            enter( pointer.trail );
            
            if ( node.isPainted() ) {
              var wrapper = wrapperStack[wrapperStack.length-1];
              
              // TODO: consider just passing the wrapper. state not needed (for now), context easily accessible
              node.paintCanvas( wrapper );
            }
            
            // TODO: restricted bounds rendering, and possibly generalize depthFirstUntil
            // var children = node._children;
            
            // check if we need to filter the children we render, and ignore nodes with few children (but allow 2, since that may prevent branches)
            // if ( state.childRestrictedBounds && children.length > 1 ) {
            //   var localRestrictedBounds = node.globalToLocalBounds( state.childRestrictedBounds );
              
            //   // don't filter if every child is inside the bounds
            //   if ( !localRestrictedBounds.containsBounds( node.parentToLocalBounds( node._bounds ) ) ) {
            //     children = node.getChildrenWithinBounds( localRestrictedBounds );
            //   }
            // }
            
            // _.each( children, function( child ) {
            //   fullRender( child, state );
            // } );
          } else {
            // not visible, so don't render the entire subtree
            return true;
          }
        } else {
          if ( invisibleCount === 0 ) {
            exit( pointer.trail );
          }
          
          invisibleCount -= node.isVisible() ? 0 : 1;
        }
        
      }, false ); // include endpoints (for now)
      
      // then walk the state back so we don't muck up any context saving that is going on, similar to how we walked it at the start
      // if the pointer is 'before' the node, call exitState since it called enterState inside the loop on it
      // if the pointer is 'after' the node, don't call its exitState since this was already done
      boundaryTrail = endPointer.trail.copy();
      var endWalkLength = endPointer.trail.length - ( endPointer.isAfter ? 1 : 0 );
      for ( i = endWalkLength - 1; i >= 0; i-- ) {
        var endNode = endPointer.trail.nodes[i];
        invisibleCount -= endNode.isVisible() ? 0 : 1;
        
        if ( invisibleCount === 0 ) {
          // walk back the state
          exit( boundaryTrail );
        }
        
        boundaryTrail.removeDescendant();
      }
    },
    
    dispose: function() {
      Layer.prototype.dispose.call( this );
      
      this.canvas.parentNode.removeChild( this.canvas );
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      matrix.canvasAppendTransform( this.context );
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      Layer.prototype.reindex.call( this, zIndex );
      
      if ( this.zIndex !== zIndex ) {
        this.canvas.style.zIndex = zIndex;
        this.zIndex = zIndex;
      }
      return zIndex + 1;
    },
    
    pushClipShape: function( shape ) {
      // store the current state, since browser support for context.resetClip() is not yet in the stable browser versions
      this.context.save();
      
      this.writeClipShape( shape );
    },
    
    popClipShape: function() {
      this.context.restore();
    },
    
    // canvas-specific
    writeClipShape: function( shape ) {
      // set up the clipping
      this.context.beginPath();
      shape.writeToContext( this.context );
      this.context.clip();
    },
    
    clearGlobalBounds: function( bounds ) {
      if ( !bounds.isEmpty() ) {
        this.context.save();
        this.context.setTransform( this.backingScale, 0, 0, this.backingScale, 0, 0 );
        this.context.clearRect( bounds.getX(), bounds.getY(), bounds.getWidth(), bounds.getHeight() );
        // use this for debugging cleared (dirty) regions for now
        // this.context.fillStyle = '#' + Math.floor( Math.random() * 0xffffff ).toString( 16 );
        // this.context.fillRect( bounds.x, bounds.y, bounds.width, bounds.height );
        this.context.restore();
      }
    },
    
    getSVGString: function() {
      return '<image xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="' + this.canvas.toDataURL() + '" x="0" y="0" height="' + this.canvas.height + 'px" width="' + this.canvas.width + 'px"/>';
    },
    
    // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
    renderToCanvas: function( canvas, context, delayCounts ) {
      context.drawImage( this.canvas, 0, 0 );
    },
    
    addNodeFromTrail: function( trail ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' addNodeFromTrail: ' + trail.toString() );
      Layer.prototype.addNodeFromTrail.call( this, trail );
      
      // since the node's getBounds() are in the parent coordinate frame, we peel off the last node to get the correct (relevant) transform
      // TODO: more efficient way of getting this transform?
      this.canvasMarkLocalBounds( trail.lastNode().getBounds(), trail.slice( 0, trail.length - 1 ) );
      
      if ( trail.lastNode().boundsInaccurate ) {
        this.boundlessCount++;
      }
    },
    
    removeNodeFromTrail: function( trail ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' removeNodeFromTrail: ' + trail.toString() );
      Layer.prototype.removeNodeFromTrail.call( this, trail );
      
      // since the node's getBounds() are in the parent coordinate frame, we peel off the last node to get the correct (relevant) transform
      // TODO: more efficient way of getting this transform?
      this.canvasMarkLocalBounds( trail.lastNode().getBounds(), trail.slice( 0, trail.length - 1 ) );
      
      if ( trail.lastNode().boundsInaccurate ) {
        this.boundlessCount--;
      }
    },
    
    canUseDirtyRegions: function() {
      assert && assert( this.boundlessCount >= 0 );
      return this.boundlessCount === 0;
    },
    
    // NOTE: for performance, we will mutate the bounds passed in (they are almost assuredly from the local or parent bounds functions)
    canvasMarkGlobalBounds: function( globalBounds ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' canvasMarkGlobalBounds: ' + globalBounds.toString() );
      assert && assert( globalBounds.isEmpty() || globalBounds.isFinite(), 'Infinite (non-empty) dirty bounds passed to canvasMarkGlobalBounds' );
      
      // TODO: for performance, consider more than just a single dirty bounding box
      this.dirtyBounds = this.dirtyBounds.union( globalBounds.dilate( 1 ).roundOut() );
    },
    
    canvasMarkLocalBounds: function( localBounds, trail ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' canvasMarkLocalBounds: ' + localBounds.toString() + ' on ' + trail.toString() );
      this.canvasMarkGlobalBounds( trail.localToGlobalBounds( localBounds ) );
    },
    
    canvasMarkParentBounds: function( parentBounds, trail ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' canvasMarkParentBounds: ' + parentBounds.toString() + ' on ' + trail.toString() );
      this.canvasMarkGlobalBounds( trail.parentToGlobalBounds( parentBounds ) );
    },
    
    canvasMarkSelf: function( instance ) {
      this.canvasMarkLocalBounds( instance.getNode().getSelfBounds(), instance.trail );
    },
    
    canvasMarkSubtree: function( instance ) {
      this.canvasMarkParentBounds( instance.getNode().getBounds(), instance.trail );
    },
    
    getName: function() {
      return 'canvas';
    },
    
    /*---------------------------------------------------------------------------*
    * Events from Instances
    *----------------------------------------------------------------------------*/
    
    notifyVisibilityChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' notifyVisibilityChange: ' + instance.trail.toString() );
      // old paint taken care of in notifyBeforeSubtreeChange()
      
      if ( instance.trail.isVisible() ) {
        this.canvasMarkSubtree( instance );
      }
    },
    
    notifyOpacityChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' notifyOpacityChange: ' + instance.trail.toString() );
      // old paint taken care of in notifyBeforeSubtreeChange()
      
      this.canvasMarkSubtree( instance );
    },
    
    // only a painted trail under this layer
    notifyBeforeSelfChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' notifyBeforeSelfChange: ' + instance.trail.toString() );
      this.canvasMarkSelf( instance );
    },
    
    notifyBeforeSubtreeChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' notifyBeforeSubtreeChange: ' + instance.trail.toString() );
      this.canvasMarkSubtree( instance );
    },
    
    // only a painted trail under this layer
    notifyDirtySelfPaint: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' notifyDirtySelfPaint: ' + instance.trail.toString() );
      this.canvasMarkSelf( instance );
    },
    
    notifyDirtySubtreePaint: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' notifyDirtySubtreePaint: ' + instance.trail.toString() );
      this.canvasMarkSubtree( instance );
    },
    
    notifyTransformChange: function( instance ) {
      // sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' notifyTransformChange: ' + instance.trail.toString() );
      // TODO: how best to mark this so if there are multiple 'movements' we don't get called more than needed?
      // this.canvasMarkSubtree( instance );
    },
    
    // only a painted trail under this layer (for now)
    notifyBoundsAccuracyChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'CanvasLayer #' + this.id + ' notifyBoundsAccuracyChange: ' + instance.trail.toString() );
      
      if ( instance.node.boundsInaccurate ) {
        this.boundlessCount++;
      } else {
        this.boundlessCount--;
      }
    }
  } );
  
  return CanvasLayer;
} );


