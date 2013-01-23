// Copyright 2002-2012, University of Colorado

/**
 * DOM nodes. Currently lightweight handling
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};
phet.scene.nodes = phet.scene.nodes || {};

(function(){
    phet.scene.nodes.DOM = function( node ) {
        phet.scene.Node.call( this );
        this.node = node;
        
        this.invalidateDOM();
        
        // can this node be interacted with? if set to true, it will not be layered like normal, but will be placed on top
        this.interactive = false;
    };
    var DOM = phet.scene.nodes.DOM;
    
    DOM.prototype = Object.create( phet.scene.Node.prototype );
    DOM.prototype.constructor = DOM;
    
    DOM.prototype.invalidateDOM = function() {
        // TODO: do we need to reset the CSS transform to get the proper bounds?
        
        // TODO: reset with the proper bounds here
        this.invalidateSelf( phet.math.Bounds2.NOTHING );
    },
    
    DOM.prototype.renderSelf = function( state ) {
        // something of the form matrix(...) as a String, with a Z translation to trigger hardware acceleration (hopefully)
        var cssTransform = 'translateZ(0) ' + state.transform.getMatrix().cssTransform();
        
        $( this.node ).css( {
            '-webkit-transform': cssTransform,
            '-moz-transform': cssTransform,
            '-ms-transform': cssTransform,
            '-o-transform': cssTransform,
            'transform': cssTransform,
            'transform-origin': 'top left', // at the origin of the component. consider 0px 0px instead. Critical, since otherwise this defaults to 50% 50%!!! see https://developer.mozilla.org/en-US/docs/CSS/transform-origin
            '-ms-transform-origin': 'top left',
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
    };
})();


