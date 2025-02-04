// Copyright 2025, University of Colorado Boulder

/**
 * Takes the element.content from himalaya, unescapes HTML entities, and applies the proper directional tags.
 *
 * See https://github.com/phetsims/scenery-phet/issues/315
 *
 * Extracted to reduce circular dependencies.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export const richTextContentToString = ( content: string, isLTR?: boolean ): string => {
  // @ts-expect-error - we should get a string from this
  const unescapedContent: string = he.decode( content );

  return isLTR === undefined ? unescapedContent : `${isLTR ? '\u202a' : '\u202b'}${unescapedContent}\u202c`;
};