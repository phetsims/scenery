// Copyright 2002-2012, University of Colorado

/**
 * A Canvas-backed layer in the scene graph. Each layer handles dirty-region handling separately,
 * and corresponds to a single canvas / svg element / DOM element in the main container.
 * Importantly, it does not contain rendered content from a subtree of the main
 * scene graph. It only will render a contiguous block of nodes visited in a depth-first
 * manner.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  var Bounds2 = phet.math.Bounds2;
  
  // assumes main is wrapped with JQuery
  scenery.CanvasLayer = function( args ) {
    scenery.Layer.call( this, args );
    
    var canvas = document.createElement( 'canvas' );
    canvas.width = this.main.width();
    canvas.height = this.main.height();
    $( canvas ).css( 'position', 'absolute' );
    
    // add this layer on top (importantly, the constructors of the layers are called in order)
    this.main.append( canvas );
    
    this.canvas = canvas;
    // this.context = new scenery.DebugContext( phet.canvas.initCanvas( canvas ) );
    this.context = phet.canvas.initCanvas( canvas );
    this.scene = args.scene;
    
    // workaround for Chrome (WebKit) miterLimit bug: https://bugs.webkit.org/show_bug.cgi?id=108763
    this.context.miterLimit = 20;
    this.context.miterLimit = 10;
    
    this.isCanvasLayer = true;
    
    this.resetStyles();
  };
  
  var CanvasLayer = scenery.CanvasLayer;
  
  CanvasLayer.prototype = _.extend( {}, scenery.Layer.prototype, {
    constructor: CanvasLayer,
    
    // function fullRender( node, state ) {
    //   node.enterState( state );
      
    //   if ( node._visible ) {
    //     node.renderSelf( state );
        
    //     var children = node.children;
        
    //     // check if we need to filter the children we render, and ignore nodes with few children (but allow 2, since that may prevent branches)
    //     if ( state.childRestrictedBounds && children.length > 1 ) {
    //       var localRestrictedBounds = node.globalToLocalBounds( state.childRestrictedBounds );
          
    //       // don't filter if every child is inside the bounds
    //       if ( !localRestrictedBounds.containsBounds( node.parentToLocalBounds( node._bounds ) ) ) {
    //         children = node.getChildrenWithinBounds( localRestrictedBounds );
    //       }
    //     }
        
    //     _.each( children, function( child ) {
    //       fullRender( child, state );
    //     } );
    //   }
      
    //   node.exitState( state );
    // }
    
    render: function( state ) {
      state.layer = this;
      var context = this.context;
      var layer = this;
      
      // if there are no clip shapes right now, we can skip the save/restore, since it will happen on the push/pop of clip shapes
      var needsStateRestoration = state.clipShapes.length > 0;
      
      // first, switch to an identity matrix so we can apply the global coordinate clipping shapes
      context.setTransform( 1, 0, 0, 1, 0, 0 );
      
      if ( needsStateRestoration ) {
        context.save();
      }
      
      _.each( state.clipShapes, function( shape ) {
        layer.writeClipShape( shape );
      } );
      
      // set the context's transform to the current transformation matrix (after we apply the clipping shapes in the global bounds)
      state.transform.getMatrix().canvasSetTransform( this.context );
      
      // reset the internal styles so they match the defaults that should be present
      this.resetStyles();
      
      // dirty bounds (clear, possibly set restricted bounds and handling for that)
      // visibility checks
      
      if ( !window.thisBetterNotExistItIsJustBecauseClosureWarnsAboutUnreachableCode ) {
        throw new Error( 'CanvasLayer.render needs to be flushed out more. render everything here' );
      }
      
      if ( needsStateRestoration ) {
        context.restore();
      }
    },
    
    dispose: function() {
      $( this.canvas ).detach();
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      matrix.canvasAppendTransform( this.context );
    },
    
    resetStyles: function() {
      this.fillStyle = null;
      this.strokeStyle = null;
      this.lineWidth = 1;
      this.lineCap = 'butt'; // default 'butt';
      this.lineJoin = 'miter';
      this.miterLimit = 10;
      
      this.font = '10px sans-serif';
      this.textAlign = 'start';
      this.textBaseline = 'alphabetic';
      this.direction = 'inherit';
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
    
    // canvas-specific
    writeClipShape: function( shape ) {
      // set up the clipping
      this.context.beginPath();
      shape.writeToContext( this.context );
      this.context.clip();
    },
    
    popClipShape: function() {
      this.context.restore();
    },
    
    prepareBounds: function( globalBounds ) {
      // don't let the bounds of the clearing go outside of the canvas
      var clearBounds = globalBounds.intersection( new phet.math.Bounds2( 0, 0, this.canvas.width, this.canvas.height ) );
      
      if ( !clearBounds.isEmpty() ) {
        this.context.save();
        this.context.setTransform( 1, 0, 0, 1, 0, 0 );
        this.context.clearRect( clearBounds.x(), clearBounds.y(), clearBounds.width(), clearBounds.height() );
        this.context.restore();
      }
    },
    
    setFillStyle: function( style ) {
      if ( this.fillStyle !== style ) {
        this.fillStyle = style;
        this.context.fillStyle = style;
      }
    },
    
    setStrokeStyle: function( style ) {
      if ( this.strokeStyle !== style ) {
        this.strokeStyle = style;
        this.context.strokeStyle = style;
      }
    },
    
    setLineWidth: function( width ) {
      if ( this.lineWidth !== width ) {
        this.lineWidth = width;
        this.context.lineWidth = width;
      }
    },
    
    setLineCap: function( cap ) {
      if ( this.lineCap !== cap ) {
        this.lineCap = cap;
        this.context.lineCap = cap;
      }
    },
    
    setLineJoin: function( join ) {
      if ( this.lineJoin !== join ) {
        this.lineJoin = join;
        this.context.lineJoin = join;
      }
    },
    
    setFont: function( font ) {
      if ( this.font !== font ) {
        this.font = font;
        this.context.font = font;
      }
    },
    
    setTextAlign: function( textAlign ) {
      if ( this.textAlign !== textAlign ) {
        this.textAlign = textAlign;
        this.context.textAlign = textAlign;
      }
    },
    
    setTextBaseline: function( textBaseline ) {
      if ( this.textBaseline !== textBaseline ) {
        this.textBaseline = textBaseline;
        this.context.textBaseline = textBaseline;
      }
    },
    
    setDirection: function( direction ) {
      if ( this.direction !== direction ) {
        this.direction = direction;
        this.context.direction = direction;
      }
    },
    
    // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
    renderToCanvas: function( canvas, context, delayCounts ) {
      context.drawImage( this.canvas, 0, 0 );
    },
    
    isDirty: function() {
      return !this.dirtyBounds.isEmpty();
    },
    
    markDirtyRegion: function( bounds ) {
      // TODO: for performance, consider more than just a single dirty bounding box
      this.dirtyBounds = this.dirtyBounds.union( bounds.dilated( 1 ).roundedOut() );
    },
    
    resetDirtyRegions: function() {
      this.dirtyBounds = Bounds2.NOTHING;
    },
    
    prepareDirtyRegions: function() {
      this.prepareBounds( this.dirtyBounds );
    },
    
    getDirtyBounds: function() {
      return this.dirtyBounds;
    }
  } );
})();


