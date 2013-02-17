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
      this.context.setTransform( 1, 0, 0, 1, 0, 0 );
      
      // reset the internal styles so they match the defaults that should be present
      this.resetStyles();
      
      var visibleDirtyBounds = args.fullRender ? scene.sceneBounds : this.dirtyBounds.intersection( scene.sceneBounds );
      this.clearGlobalBounds( visibleDirtyBounds );
      
      if ( !args.fullRender ) {
        state.pushClipShape( scenery.Shape.bounds( visibleDirtyBounds ) );
      }
      
      // dirty bounds (clear, possibly set restricted bounds and handling for that)
      // visibility checks
      this.recursiveRender( state, args );
      
      // exists for now so that we pop the necessary context state
      if ( !args.fullRender ) {
        state.popClipShape();
      }
      
      // we rendered everything, no more dirty bounds
      this.dirtyBounds = Bounds2.NOTHING;
    },
    
    recursiveRender: function( state, args ) {
      var i;
      var startPointer = this.getStartPointer();
      var endPointer = this.getEndPointer();
      
      // first, we need to walk the state up to before our pointer (as far as the recursive handling is concerned)
      // if the pointer is 'before' the node, don't call its enterState since this will be taken care of as the first step.
      // if the pointer is 'after' the node, call enterState since it will call exitState immediately inside the loop
      var startWalkLength = startPointer.trail.length - ( startPointer.isBefore ? 1 : 0 );
      for ( i = 0; i < startWalkLength; i++ ) {
        startPointer.trail.nodes[i].enterState( state );
      }
      
      startPointer.depthFirstUntil( endPointer, function( pointer ) {
        // handle render here
        
        var node = pointer.trail.lastNode();
        
        if ( pointer.isBefore ) {
          node.enterState( state );
          
          if ( node._visible ) {
            node.renderSelf( state );
            
            // TODO: restricted bounds rendering, and possibly generalize depthFirstUntil
            // var children = node.children;
            
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
          node.exitState( state );
        }
        
      }, false ); // include endpoints (for now)
      
      // then walk the state back so we don't muck up any context saving that is going on, similar to how we walked it at the start
      // if the pointer is 'before' the node, call exitState since it called enterState inside the loop on it
      // if the pointer is 'after' the node, don't call its exitState since this was already done
      var endWalkLength = endPointer.trail.length - ( endPointer.isAfter ? 1 : 0 );
      for ( i = endWalkLength - 1; i >= 0; i-- ) {
        endPointer.trail.nodes[i].exitState( state );
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
        this.context.setTransform( 1, 0, 0, 1, 0, 0 );
        this.context.clearRect( bounds.x(), bounds.y(), bounds.width(), bounds.height() );
        // use this for debugging cleared (dirty) regions for now
        // this.context.fillStyle = '#' + Math.floor( Math.random() * 0xffffff ).toString( 16 );
        // this.context.fillRect( bounds.x(), bounds.y(), bounds.width(), bounds.height() );
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
    
    markDirtyRegion: function( node, localBounds, transform, trail ) {
      var bounds = transform.transformBounds2( localBounds );
      
      // TODO: for performance, consider more than just a single dirty bounding box
      this.dirtyBounds = this.dirtyBounds.union( bounds.dilated( 1 ).roundedOut() );
    },
    
    getName: function() {
      return 'canvas';
    }
  } );
})();


