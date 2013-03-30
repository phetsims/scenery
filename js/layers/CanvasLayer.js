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
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Shape = require( 'KITE/Shape' );
  
  var Layer = require( 'SCENERY/layers/Layer' ); // uses Layer's prototype for inheritance
  require( 'SCENERY/util/CanvasContextWrapper' );
  require( 'SCENERY/util/RenderState' );
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/TrailPointer' );
  require( 'SCENERY/util/Util' );
  
  // assumes main is wrapped with JQuery
  /*
   *
   */
  scenery.CanvasLayer = function( args ) {
    Layer.call( this, args );
    
    // TODO: deprecate Scene's backing scale, and handle this on a layer-by-layer option?
    this.backingScale = args.scene.backingScale;
    if ( args.fullResolution !== undefined ) {
      this.backingScale = args.fullResolution ? scenery.Util.backingScale( document.createElement( 'canvas' ).getContext( '2d' ) ) : 1;
    }
    
    this.logicalWidth = this.$main.width();
    this.logicalHeight = this.$main.height();
    
    var canvas = document.createElement( 'canvas' );
    canvas.width = this.logicalWidth * this.backingScale;
    canvas.height = this.logicalHeight * this.backingScale;
    $( canvas ).css( 'width', this.logicalWidth );
    $( canvas ).css( 'height', this.logicalHeight );
    $( canvas ).css( 'position', 'absolute' );
    
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
  };
  var CanvasLayer = scenery.CanvasLayer;
  
  CanvasLayer.prototype = _.extend( {}, Layer.prototype, {
    constructor: CanvasLayer,
    
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
      
      // bail out quickly if possible
      if ( !args.fullRender && this.dirtyBounds.isEmpty() ) {
        return;
      }
      
      var state = new scenery.RenderState( scene );
      state.layer = this;
      
      // switch to an identity transform
      this.context.setTransform( this.backingScale, 0, 0, this.backingScale, 0, 0 );
      
      var visibleDirtyBounds = args.fullRender ? scene.sceneBounds : this.dirtyBounds.intersection( scene.sceneBounds );
      
      if ( !visibleDirtyBounds.isEmpty() ) {
        this.clearGlobalBounds( visibleDirtyBounds );
        
        if ( !args.fullRender ) {
          state.pushClipShape( Shape.bounds( visibleDirtyBounds ) );
        }
        
        // dirty bounds (clear, possibly set restricted bounds and handling for that)
        // visibility checks
        this.recursiveRender( scene, state, args );
        
        // exists for now so that we pop the necessary context state
        if ( !args.fullRender ) {
          state.popClipShape();
        }
      }
      
      // we rendered everything, no more dirty bounds
      this.dirtyBounds = Bounds2.NOTHING;
    },
    
    recursiveRender: function( scene, state, args ) {
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
        var context = canvas.getContext( '2d' );
        
        return new scenery.CanvasContextWrapper( canvas, context );
      }
      
      function enter( state, trail ) {
        trail.lastNode().enterState( state, trail );
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
        }
      }
      
      function exit( state, trail ) {
        trail.lastNode().exitState( state, trail );
        if ( requiresScratchCanvas( trail ) ) {
          var baseContext = wrapperStack[wrapperStack.length-2].context;
          var topCanvas = wrapperStack[wrapperStack.length-1].canvas;
          
          // apply necessary style transforms before painting our popped canvas onto the next canvas
          var opacityChange = trail.lastNode().getOpacity() < 1;
          if ( opacityChange ) {
            baseContext.globalAlpha = trail.lastNode().getOpacity();
          }
          
          // paint our canvas onto the level below
          baseContext.drawImage( topCanvas, 0, 0 );
          
          // reset styles
          if ( opacityChange ) {
            baseContext.globalAlpha = 1;
          }
          
          wrapperStack.pop();
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
          enter( state, boundaryTrail );
        }
      }
      
      startPointer.depthFirstUntil( endPointer, function renderPointer( pointer ) {
        // handle render here
        
        var node = pointer.trail.lastNode();
        
        if ( pointer.isBefore ) {
          invisibleCount += node.isVisible() ? 0 : 1;
          
          if ( invisibleCount === 0 ) {
            enter( state, pointer.trail );
            
            if ( node.isPainted() ) {
              var wrapper = wrapperStack[wrapperStack.length-1];
              
              // TODO: consider just passing the wrapper. state not needed (for now), context easily accessible
              node.paintCanvas( state, wrapper, wrapper.context );
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
            exit( state, pointer.trail );
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
          exit( state, boundaryTrail );
        }
        
        boundaryTrail.removeDescendant();
      }
    },
    
    dispose: function() {
      Layer.prototype.dispose.call( this );
      $( this.canvas ).detach();
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      matrix.canvasAppendTransform( this.context );
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      $( this.canvas ).css( 'z-index', zIndex );
      this.zIndex = zIndex;
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
    
    markDirtyRegion: function( args ) {
      this.internalMarkDirtyBounds( args.bounds, args.transform );
    },
    
    addNodeFromTrail: function( trail ) {
      Layer.prototype.addNodeFromTrail.call( this, trail );
      
      // since the node's getBounds() are in the parent coordinate frame, we peel off the last node to get the correct (relevant) transform
      // TODO: more efficient way of getting this transform?
      this.internalMarkDirtyBounds( trail.lastNode().getBounds(), trail.slice( 0, trail.length - 1 ).getTransform() );
    },
    
    removeNodeFromTrail: function( trail ) {
      Layer.prototype.removeNodeFromTrail.call( this, trail );
      
      // since the node's getBounds() are in the parent coordinate frame, we peel off the last node to get the correct (relevant) transform
      // TODO: more efficient way of getting this transform?
      this.internalMarkDirtyBounds( trail.lastNode().getBounds(), trail.slice( 0, trail.length - 1 ).getTransform() );
    },
    
    internalMarkDirtyBounds: function( localBounds, transform ) {
      assert && assert( localBounds.isEmpty() || localBounds.isFinite(), 'Infinite (non-empty) dirty bounds passed to internalMarkDirtyBounds' );
      var globalBounds = transform.transformBounds2( localBounds );
      
      // TODO: for performance, consider more than just a single dirty bounding box
      this.dirtyBounds = this.dirtyBounds.union( globalBounds.dilated( 1 ).roundedOut() );
    },
    
    transformChange: function( args ) {
      // currently no-op, since this is taken care of by markDirtyRegion
    },
    
    getName: function() {
      return 'canvas';
    }
  } );
  
  return CanvasLayer;
} );


