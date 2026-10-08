/** A count spelled out for headlines, such as "Thirty-two" (1 to 99). */
export function countWords(n: number): string {
  const ones = [
    '',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve',
    'thirteen',
    'fourteen',
    'fifteen',
    'sixteen',
    'seventeen',
    'eighteen',
    'nineteen',
  ];
  const tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
  if (!Number.isInteger(n) || n < 1 || n > 99) throw new Error(`countWords() takes 1 to 99, got ${n}.`);
  const w = n < 20 ? ones[n] : tens[Math.floor(n / 10)] + (n % 10 ? `-${ones[n % 10]}` : '');
  return w[0].toUpperCase() + w.slice(1);
}
