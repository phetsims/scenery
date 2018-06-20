// Copyright 2016-2017, University of Colorado Boulder

/**
 * IO type for Text
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Denzell Barnett (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var FontIO = require( 'SCENERY/util/FontIO' );
  var NodeIO = require( 'SCENERY/nodes/NodeIO' );
  var scenery = require( 'SCENERY/scenery' );

  // ifphetio
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var FunctionIO = require( 'ifphetio!PHET_IO/types/FunctionIO' );
  var NumberIO = require( 'ifphetio!PHET_IO/types/NumberIO' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var StringIO = require( 'ifphetio!PHET_IO/types/StringIO' );
  var VoidIO = require( 'ifphetio!PHET_IO/types/VoidIO' );

  /**
   * @param {Text} text
   * @param {string} phetioID
   * @constructor
   */
  function TextIO( text, phetioID ) {
    assert && assertInstanceOf( text, phet.scenery.Text );
    NodeIO.call( this, text, phetioID );
  }

  phetioInherit( NodeIO, 'TextIO', TextIO, {

    addTextChangedListener: {
      returnType: VoidIO,
      parameterTypes: [ FunctionIO( VoidIO, [ StringIO, StringIO ] ) ],
      implementation: function( listener ) {
        this.instance.on( 'text', function( oldText, newText ) {
          listener( newText, oldText );
        } );
      },
      documentation: 'Adds a listener for when the text has changed. The listener takes two arguments, the new ' +
                     'value and the previous value.'
    },

    setText: {
      returnType: VoidIO,
      parameterTypes: [ StringIO ],
      implementation: function( text ) {
        this.instance.text = text;
      },
      documentation: 'Sets the text content'
    },

    getText: {
      returnType: StringIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.text;
      },
      documentation: 'Gets the text content'
    },

    setFontOptions: {
      returnType: VoidIO,
      parameterTypes: [ FontIO ],
      implementation: function( font ) {
        this.instance.setFont( font );
      },
      documentation: 'Sets font options for this TextIO instance, e.g. {size: 16, weight: bold}. If increasing the font ' +
                     'size does not make the text size larger, you may need to increase the maxWidth of the TextIO also.'
    },

    getFontOptions: {
      returnType: FontIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.getFont();
      },
      documentation: 'Gets font options for this TextIO instance as an object'
    },

    setMaxWidth: {
      returnType: VoidIO,
      parameterTypes: [ NumberIO ],
      implementation: function( maxWidth ) {
        this.instance.setMaxWidth( maxWidth );
      },
      documentation: 'Sets the maximum width of text box. ' +
                     'If the text width exceeds maxWidth, it is scaled down to fit.'
    },

    getMaxWidth: {
      returnType: NumberIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.maxWidth;
      },
      documentation: 'Gets the maximum width of text box'
    }
  }, {
    documentation: 'Text that is displayed in the simulation.',
    events: [ 'textChanged' ]
  } );

  scenery.register( 'TextIO', TextIO );

  return TextIO;
} );
