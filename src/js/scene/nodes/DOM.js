// Copyright 2002-2012, University of Colorado

/**
 * DOM nodes. Currently lightweight handling
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    "use strict";
    
    phet.scene.DOM = function( node ) {
        phet.scene.Node.call( this );
        this.node = node;
        
        // can this node be interacted with? if set to true, it will not be layered like normal, but will be placed on top
        this.interactive = false;
        
        this.invalidateDOM();
    };
    var DOM = phet.scene.DOM;
    
    DOM.prototype = Object.create( phet.scene.Node.prototype );
    DOM.prototype.constructor = DOM;
    
    DOM.prototype.invalidateDOM = function() {
        // TODO: do we need to reset the CSS transform to get the proper bounds?
        
        // TODO: reset with the proper bounds here
        this.invalidateSelf( new phet.math.Bounds2( 0, 0, $( this.node ).width(), $( this.node ).height() ) );
    },
    
    DOM.prototype.renderSelf = function( state ) {
        // something of the form matrix(...) as a String, with a Z translation to trigger hardware acceleration (hopefully)
        var cssTransform = state.transform.getMatrix().cssTransform();
        
        $( this.node ).css( {
            '-webkit-transform': cssTransform + ' translateZ(0)', // trigger hardware acceleration if possible
            '-moz-transform': cssTransform + ' translateZ(0)', // trigger hardware acceleration if possible
            '-ms-transform': cssTransform,
            '-o-transform': cssTransform,
            'transform': cssTransform,
            'transform-origin': 'top left', // at the origin of the component. consider 0px 0px instead. Critical, since otherwise this defaults to 50% 50%!!! see https://developer.mozilla.org/en-US/docs/CSS/transform-origin
            '-ms-transform-origin': 'top left', // TODO: do we need other platform-specific transform-origin styles?
            position: 'absolute',
            left: 0,
            top: 0
        } );
        
        var layer = ( state.isDOMState() && !this.interactive ) ? state.layer : state.scene.getUILayer();
        
        var container = layer.getContainer();
        
        if( this.node.parentNode !== container ) {
            // TODO: correct layering of DOM nodes inside the container
            $( container ).append( this.node );
        }
        
        // TODO: this will only reset bounds after rendering the first time. this might never render in the first place?
        this.invalidateDOM();
    };
})();


