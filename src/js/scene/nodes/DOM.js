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
    
    phet.scene.DOM = function( element, params ) {
        phet.scene.Node.call( this, params );
        this._element = element;
        
        // can this node be interacted with? if set to true, it will not be layered like normal, but will be placed on top
        this._interactive = false;
        
        this.invalidateDOM();
    };
    var DOM = phet.scene.DOM;
    
    DOM.prototype = Object.create( phet.scene.Node.prototype );
    DOM.prototype.constructor = DOM;
    
    DOM.prototype.invalidateDOM = function() {
        // TODO: do we need to reset the CSS transform to get the proper bounds?
        
        // TODO: reset with the proper bounds here
        this.invalidateSelf( new phet.math.Bounds2( 0, 0, $( this._element ).width(), $( this._element ).height() ) );
    },
    
    DOM.prototype.renderSelf = function( state ) {
        // something of the form matrix(...) as a String, with a Z translation to trigger hardware acceleration (hopefully)
        var cssTransform = state.transform.getMatrix().cssTransform();
        
        $( this._element ).css( {
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
        
        var layer = ( state.isDOMState() && !this._interactive ) ? state.layer : state.scene.getUILayer();
        
        var container = layer.getContainer();
        
        if( this._element.parentNode !== container ) {
            // TODO: correct layering of DOM nodes inside the container
            $( container ).append( this._element );
        }
        
        // TODO: this will only reset bounds after rendering the first time. this might never render in the first place?
        this.invalidateDOM();
    };
    
    DOM.prototype.setElement = function( element ) {
        this._element = element;
        
        // TODO: bounds issue, since this will probably set to empty bounds and thus a repaint may not draw over it
        this.invalidateDOM();
        
        return this; // allow chaining
    };
    
    DOM.prototype.getElement = function() {
        return this._element;
    };
    
    DOM.prototype.setInteractive = function( interactive ) {
        this._interactive = interactive;
        
        this.invalidatePaint();
        
        return this; // allow chaining
    };
    
    DOM.prototype.isInteractive = function() {
        return this._interactive;
    };
    
    DOM.prototype.mutate = function( params ) {
        var setterKeys = [ 'element', 'interactive' ];
        
        var node = this;
        
        _.each( setterKeys, function( key ) {
            if( params[key] !== undefined ) {
                node[key] = params[key];
            }
        } );
        
        phet.scene.Node.prototype.mutate.call( this, params );
    };
    
    Object.defineProperty( DOM.prototype, 'element', { set: DOM.prototype.setElement, get: DOM.prototype.getElement } );
    Object.defineProperty( DOM.prototype, 'interactive', { set: DOM.prototype.setInteractive, get: DOM.prototype.isInteractive } );
    
})();


