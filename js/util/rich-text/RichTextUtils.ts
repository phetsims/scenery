// Copyright 2023, University of Colorado Boulder

/**
 * Utilities and globals to support RichText
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import { scenery, Text } from '../../imports.js';

// Types for Himalaya
export type HimalayaAttribute = {
  key: string;
  value?: string;
};

export type HimalayaNode = {
  type: 'element' | 'comment' | 'text';
  innerContent: string;
};

export type HimalayaElementNode = {
  type: 'element';
  tagName: string;
  children: HimalayaNode[];
  attributes: HimalayaAttribute[];
  innerContent?: string; // Is this in the generated stuff? Do we just override this? Unclear
} & HimalayaNode;

export const isHimalayaElementNode = ( node: HimalayaNode ): node is HimalayaElementNode => node.type.toLowerCase() === 'element';

export type HimalayaTextNode = {
  type: 'text';
  content: string;
} & HimalayaNode;

export const isHimalayaTextNode = ( node: HimalayaNode ): node is HimalayaTextNode => node.type.toLowerCase() === 'text';

const RichTextUtils = {
  // We need to do some font-size tests, so we have a Text for that.
  scratchText: new Text( '' ),

  // Get the attribute value from an element. Return null if that attribute isn't on the element.
  himalayaGetAttribute( attribute: string, element: HimalayaElementNode | null ): string | null {
    if ( !element ) {
      return null;
    }
    const attributeObject = _.find( element.attributes, x => x.key === attribute );
    if ( !attributeObject ) {
      return null;
    }
    return attributeObject.value || null;
  },

  // Turn a string of style like "font-sie:6; font-weight:6; favorite-number:6" into a may of style key/values (trimmed of whitespace)
  himalayaStyleStringToMap( styleString: string ): Record<string, string> {
    const styleElements = styleString.split( ';' );
    const styleMap: Record<string, string> = {};
    styleElements.forEach( styleKeyValue => {
      if ( styleKeyValue.length > 0 ) {
        const keyValueTuple = styleKeyValue.split( ':' );
        assert && assert( keyValueTuple.length === 2, 'too many colons' );
        styleMap[ keyValueTuple[ 0 ].trim() ] = keyValueTuple[ 1 ].trim();
      }
    } );
    return styleMap;
  }
};

export default RichTextUtils;

scenery.register( 'RichTextUtils', RichTextUtils );
