// Copyright 2002-2012, University of Colorado

/**
 * A DOM-based layer in the scene graph. Each layer handles dirty-region handling separately,
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
    phet.scene.DOMLayer = function( args ) {
        var main = args.main;
        this.main = main;
        
        this.div = document.createElement( 'div' );
        $( this.div ).width( main.width() );
        $( this.div ).height( main.height() );
        $( this.div ).css( 'position', 'absolute' );
        main.append( this.div );
        
        this.scene = args.scene;
        
        this.dirtyBounds = Bounds2.NOTHING;
        
        this.isDOMLayer = true;
        
        // references to surrounding layers, filled by rebuildLayers
        this.nextLayer = null;
        this.previousLayer = null;
    };
    
    var DOMLayer = phet.scene.DOMLayer;
    
    DOMLayer.prototype = {
        constructor: DOMLayer,
        
        // called when rendering switches to this layer
        initialize: function( renderState ) {
            // TODO: initialize transform
            // TODO: clipping
        },
        
        // called when rendering switches away from this layer
        cooldown: function() {
            
        },
        
        // called if it needs to be added back to the main element after elements are removed
        recreate: function() {
            this.main.append( this.div );
        },
        
        isDirty: function() {
            return false;
        },
        
        // TODO: consider a stack-based model for transforms?
        applyTransformationMatrix: function( matrix ) {
            
        },
        
        getContainer: function() {
            return this.div;
        },
        
        // returns next zIndex in place. allows layers to take up more than one single zIndex
        reindex: function( zIndex ) {
            $( this.div ).css( 'z-index', zIndex );
            this.zIndex = zIndex;
            return zIndex + 1;
        },
        
        pushClipShape: function( shape ) {
            // TODO: clipping
        },
        
        popClipShape: function() {
            // TODO: clipping
        },
        
        markDirtyRegion: function( bounds ) {
            // null op, always clean
        },
        
        resetDirtyRegions: function() {
            this.dirtyBounds = Bounds2.NOTHING;
        },
        
        prepareBounds: function( globalBounds ) {
            // null op
        },
        
        prepareDirtyRegions: function() {
            this.prepareBounds( this.dirtyBounds );
        },
        
        getDirtyBounds: function() {
            return this.dirtyBounds;
        },
        
        // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
        renderToCanvas: function( canvas, context, delayCounts ) {
            var data = "<svg xmlns='http://www.w3.org/2000/svg' width='" + this.main.width() + "' height='" + this.main.height() + "'>" +
                "<foreignObject width='100%' height='100%'>" +
                $( this.div ).html() + 
                "</foreignObject></svg>";
            
            var DOMURL = self.URL || self.webkitURL || self;
            var img = new Image();
            var svg = new Blob( [ data ] , { type: "image/svg+xml;charset=utf-8" } );
            var url = DOMURL.createObjectURL( svg );
            delayCounts.increment();
            img.onload = function() {
                context.drawImage( img, 0, 0 );
                // TODO: this loading is delayed!!! ... figure out a solution to potentially delay?
                DOMURL.revokeObjectURL( url );
                delayCounts.decrement();
            };
            img.src = url;
        }
    };
})();


