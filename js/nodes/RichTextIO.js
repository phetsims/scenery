// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO Type for RichText
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import PropertyIO from '../../../axon/js/PropertyIO.js';
import merge from '../../../phet-core/js/merge.js';
import IOType from '../../../tandem/js/types/IOType.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import scenery from '../scenery.js';
import NodeProperty from '../util/NodeProperty.js';
import NodeIO from './NodeIO.js';

const RichTextIO = new IOType( 'RichTextIO', {
  valueType: scenery.RichText,
  supertype: NodeIO,
  documentation: 'The tandem IO Type for the scenery RichText node',

  // https://github.com/phetsims/tandem/issues/211 and dispose
  createWrapper( richText, phetioID ) {

    // this uses a sub Property adapter as described in https://github.com/phetsims/phet-io/issues/1326
    const textProperty = new NodeProperty( richText, richText.textProperty, 'text', merge( {

      // pick the following values from the parent Node
      phetioReadOnly: richText.phetioReadOnly,
      phetioType: PropertyIO( StringIO ),

      tandem: richText.tandem.createTandem( 'textProperty' ),
      phetioDocumentation: 'Property for the displayed text'
    }, richText.phetioComponentOptions, richText.phetioComponentOptions.textProperty ) );

    return {
      phetioObject: richText,
      phetioID: phetioID,
      dispose: () => textProperty.dispose()
    };
  }
} );

scenery.register( 'RichTextIO', RichTextIO );
export default RichTextIO;