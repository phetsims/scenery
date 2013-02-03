// Copyright 2002-2012, University of Colorado

/**
 * A Canvas-backed layer in the scene graph. Each layer handles dirty-region handling separately,
 * and corresponds to a single canvas / svg element / DOM element in the main container.
 * Importantly, it does not contain rendered content from a subtree of the main
 * scene graph. It only will render a contiguous block of nodes visited in a depth-first
 * manner.
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    "use strict";
    
    var Bounds2 = phet.math.Bounds2;
    
    // assumes main is wrapped with JQuery
    phet.scene.CanvasLayer = function( args ) {
        var main = args.main;
        var canvas = document.createElement( 'canvas' );
        canvas.width = main.width();
        canvas.height = main.height();
        $( canvas ).css( 'position', 'absolute' );
        
        // add this layer on top (importantly, the constructors of the layers are called in order)
        main.append( canvas );
        
        this.canvas = canvas;
        // this.context = new phet.scene.DebugContext( phet.canvas.initCanvas( canvas ) );
        this.context = phet.canvas.initCanvas( canvas );
        this.scene = args.scene;
        
        // workaround for Chrome (WebKit) miterLimit bug: https://bugs.webkit.org/show_bug.cgi?id=108763
        this.context.miterLimit = 20;
        this.context.miterLimit = 10;
        
        this.isCanvasLayer = true;
        
        // initialize to fully dirty so we draw everything the first time
        // bounds in global coordinate frame
        this.dirtyBounds = Bounds2.EVERYTHING;
        
        this.resetStyles();
        
        // filled in after construction by an external source (currently Scene.rebuildLayers).
        this.startNode = null;
        this.endNode = null;
        
        // references to surrounding layers, filled by rebuildLayers
        this.nextLayer = null;
        this.previousLayer = null;
    };
    
    var CanvasLayer = phet.scene.CanvasLayer;
    
    CanvasLayer.prototype = {
        constructor: CanvasLayer,
        
        // called when rendering switches to this layer
        initialize: function( renderState ) {
            // first, switch to an identity matrix so we can apply the global coordinate clipping shapes
            this.context.setTransform( 1, 0, 0, 1, 0, 0 );
            
            // save now, so that we can clear the clipping shapes later
            this.context.save();
            
            var context = this.context;
            
            // apply clipping shapes in the global coordinate frame
            _.each( renderState.clipShapes, function( shape ) {
                context.beginPath();
                shape.writeToContext( context );
                context.clip();
            } );
            
            // set the context's transform to the current transformation matrix
            var matrix = renderState.transform.getMatrix().canvasSetTransform( this.context );
            
            // reset the styles so that they are re-done
            this.resetStyles();
        },
        
        // called when rendering switches away from this layer
        cooldown: function() {
            this.context.restore();
        },
        
        isDirty: function() {
            return !this.dirtyBounds.isEmpty();
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
        
        // called if it needs to be added back to the main element after elements are removed
        recreate: function() {
            this.main.append( this.canvas );
        },
        
        pushClipShape: function( shape ) {
            // store the current state, since browser support for context.resetClip() is not yet in the stable browser versions
            this.context.save();
            
            // set up the clipping
            this.context.beginPath();
            shape.writeToContext( this.context );
            this.context.clip();
        },
        
        popClipShape: function() {
            this.context.restore();
        },
        
        markDirtyRegion: function( bounds ) {
            // TODO: for performance, consider more than just a single dirty bounding box
            this.dirtyBounds = this.dirtyBounds.union( bounds.dilated( 1 ).roundedOut() );
        },
        
        resetDirtyRegions: function() {
            this.dirtyBounds = Bounds2.NOTHING;
        },
        
        prepareBounds: function( globalBounds ) {
            // don't let the bounds of the clearing go outside of the canvas
            var clearBounds = globalBounds.intersection( new phet.math.Bounds2( 0, 0, this.canvas.width, this.canvas.height ) );
            
            if( !clearBounds.isEmpty() ) {
                this.context.save();
                this.context.setTransform( 1, 0, 0, 1, 0, 0 );
                this.context.clearRect( clearBounds.x(), clearBounds.y(), clearBounds.width(), clearBounds.height() );
                this.context.restore();
            }
        },
        
        prepareDirtyRegions: function() {
            this.prepareBounds( this.dirtyBounds );
        },
        
        getDirtyBounds: function() {
            return this.dirtyBounds;
        },
        
        setFillStyle: function( style ) {
            if( this.fillStyle !== style ) {
                this.fillStyle = style;
                this.context.fillStyle = style;
            }
        },
        
        setStrokeStyle: function( style ) {
            if( this.strokeStyle !== style ) {
                this.strokeStyle = style;
                this.context.strokeStyle = style;
            }
        },
        
        setLineWidth: function( width ) {
            if( this.lineWidth !== width ) {
                this.lineWidth = width;
                this.context.lineWidth = width;
            }
        },
        
        setLineCap: function( cap ) {
            if( this.lineCap !== cap ) {
                this.lineCap = cap;
                this.context.lineCap = cap;
            }
        },
        
        setLineJoin: function( join ) {
            if( this.lineJoin !== join ) {
                this.lineJoin = join;
                this.context.lineJoin = join;
            }
        },
        
        setFont: function( font ) {
            if( this.font !== font ) {
                this.font = font;
                this.context.font = font;
            }
        },
        
        setTextAlign: function( textAlign ) {
            if( this.textAlign !== textAlign ) {
                this.textAlign = textAlign;
                this.context.textAlign = textAlign;
            }
        },
        
        setTextBaseline: function( textBaseline ) {
            if( this.textBaseline !== textBaseline ) {
                this.textBaseline = textBaseline;
                this.context.textBaseline = textBaseline;
            }
        },
        
        setDirection: function( direction ) {
            if( this.direction !== direction ) {
                this.direction = direction;
                this.context.direction = direction;
            }
        },
        
        // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
        renderToCanvas: function( canvas, context, delayCounts ) {
            context.drawImage( this.canvas, 0, 0 );
        }
    };
})();


